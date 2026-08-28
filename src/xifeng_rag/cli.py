from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xifeng_rag import __version__
from xifeng_rag.chunking import (
    CodexCliClient,
    OllamaClient,
    OpenAICompatibleClient,
    approve_manifest,
    export_review,
    propose_chunks,
    resolve_cleanup,
)
from xifeng_rag.config import Settings, load_settings
from xifeng_rag.context_assembly import assemble_context
from xifeng_rag.corpus import discover_markdown, scan_corpus
from xifeng_rag.corpus_ui import run_corpus_dashboard
from xifeng_rag.domain import (
    AssemblyResult,
    AssemblyStatus,
    BuildStatus,
    ContextAssemblyConfigSnapshot,
    ProposalStatus,
    RetrievalMode,
    VersionStatus,
)
from xifeng_rag.evaluation import run_evaluation, run_frozen_evaluation
from xifeng_rag.evaluation_ui import run_evaluation_server
from xifeng_rag.generation import (
    AnswerGenerationResult,
    AnswerStatus,
    DeepSeekClient,
    EvidenceBoundGenerator,
    GeneratedAnswer,
    SupplementPolicy,
    cited_evidence,
    render_answer,
)
from xifeng_rag.indexing import (
    IndexBuildError,
    SentenceTransformersEmbeddingProvider,
    build_hybrid_index,
    build_index,
    record_build_failure,
    search,
)
from xifeng_rag.reranking import SentenceTransformersReranker
from xifeng_rag.review_ui import IndexBuilder, run_review_server
from xifeng_rag.storage import (
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    IndexBuildRow,
    ParagraphRow,
)

app = typer.Typer(help="西风文章 RAG 研究台 V0", no_args_is_help=True)
corpus_app = typer.Typer(help="语料扫描与清洗")
chunks_app = typer.Typer(help="话题分块与审核")
index_app = typer.Typer(help="BM25 索引")
eval_app = typer.Typer(help="检索评测")
generator_app = typer.Typer(help="外部回答 Generator 诊断")
app.add_typer(corpus_app, name="corpus")
app.add_typer(chunks_app, name="chunks")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(generator_app, name="generator")


@dataclass
class Runtime:
    settings: Settings
    database: Database


def _embedding_provider(settings: Settings) -> SentenceTransformersEmbeddingProvider:
    if settings.embedding.provider != "sentence-transformers":
        raise ValueError(f"不支持的 Embedding provider：{settings.embedding.provider}")
    return SentenceTransformersEmbeddingProvider(
        settings.embedding.model_id,
        revision=settings.embedding.model_revision,
        cache_dir=settings.embedding_cache_dir,
        device=settings.embedding.device,
        batch_size=settings.embedding.batch_size,
        normalize_embeddings=settings.embedding.normalize_embeddings,
        query_instruction=settings.embedding.query_instruction,
    )


def _reranker_provider(settings: Settings) -> SentenceTransformersReranker:
    if settings.reranker.provider != "sentence-transformers":
        raise ValueError(f"不支持的 Reranker provider：{settings.reranker.provider}")
    return SentenceTransformersReranker(
        settings.reranker.model_id,
        revision=settings.reranker.model_revision,
        cache_dir=settings.reranker_cache_dir,
        device=settings.reranker.device,
        batch_size=settings.reranker.batch_size,
        max_pair_tokens=settings.reranker.max_pair_tokens,
    )


def _hybrid_index_builder(settings: Settings) -> IndexBuilder:
    def builder(session: Session, index_dir: Path):
        provider = _embedding_provider(settings)
        try:
            return build_hybrid_index(
                session,
                index_dir,
                provider,
                bm25_candidate_k=settings.retrieval.bm25_candidate_k,
                dense_candidate_k=settings.retrieval.dense_candidate_k,
                rrf_k=settings.retrieval.rrf_k,
            )
        finally:
            provider.close()

    return builder


def _resolved_retrieval_mode(
    runtime: Runtime, requested: RetrievalMode | None, build_id: str | None
) -> RetrievalMode:
    if requested is not None:
        return requested
    with runtime.database.session() as session:
        build = (
            session.get(IndexBuildRow, build_id)
            if build_id is not None
            else session.scalar(
                select(IndexBuildRow).where(
                    IndexBuildRow.active.is_(True),
                    IndexBuildRow.status == BuildStatus.ACTIVE.value,
                )
            )
        )
    return (
        RetrievalMode.HYBRID
        if build is not None and (Path(build.path) / "qdrant").is_dir()
        else RetrievalMode.BM25
    )


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="YAML 配置文件；环境变量使用 XIFENG_ 前缀"),
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="显示版本"),
    ] = False,
) -> None:
    del version
    try:
        settings = load_settings(config)
        database = Database(settings.database_path)
        database.initialize()
        ctx.obj = Runtime(settings=settings, database=database)
    except Exception as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(2) from exc


def _runtime(ctx: typer.Context) -> Runtime:
    runtime = ctx.find_object(Runtime)
    if runtime is None:
        raise RuntimeError("CLI 运行上下文未初始化")
    return runtime


def _abort(exc: Exception) -> None:
    typer.echo(f"错误：{exc}", err=True)
    raise typer.Exit(1) from exc


def _assembly_config(settings: Settings) -> ContextAssemblyConfigSnapshot:
    return ContextAssemblyConfigSnapshot.model_validate(
        settings.context_assembly.model_dump()
    )


def _echo_assembly(result: AssemblyResult) -> None:
    typer.echo(
        f"组装状态：{result.status.value}；输入 {result.usage.input_candidate_count}，"
        f"入选 {result.usage.selected_count}，排除 {result.usage.excluded_count}；"
        f"渲染 {result.usage.rendered_context_chars} 字符"
    )
    for item in result.evidence_items:
        typer.echo(
            f"{item.citation_id}. {item.article_title} {item.article_version} "
            f"¶{item.selected_paragraph_range[0]}–{item.selected_paragraph_range[1]} "
            f"（输入排名 {item.input_rank}）"
        )
        if result.config_snapshot.include_ai_metadata:
            typer.echo(f"   AI 话题：{item.topic_title or '—'}")
            typer.echo(f"   AI 概括：{item.context_summary or '—'}")
        typer.echo(f"   原文：{item.original_text}")
    for item in result.excluded_candidates:
        typer.echo(
            f"排除：rank={item.original_rank} chunk={item.source_chunk_id} "
            f"reason={item.reason.value} rendered_chars={item.rendered_chars}"
        )
    typer.echo(f"上下文 SHA-256：{result.rendered_context_sha256}")


def _answer_json(result: AnswerGenerationResult, context: AssemblyResult) -> str:
    sources = []
    for item in cited_evidence(result.answer, context):
        excerpt = item.original_text[:240].replace("\n", " ")
        if len(item.original_text) > 240:
            excerpt += "…"
        sources.append(
            {
                "citation_id": item.citation_id,
                "article_title": item.article_title,
                "article_version": item.article_version,
                "paragraph_range": list(item.selected_paragraph_range),
                "original_text_excerpt": excerpt,
            }
        )
    return json.dumps(
        {
            **result.model_dump(mode="json"),
            "rendered_answer": render_answer(result.answer, context),
            "sources": sources,
        },
        ensure_ascii=False,
        indent=2,
    )


@app.command("init")
def initialize(ctx: typer.Context) -> None:
    """创建本地运行目录和配置模板。"""
    runtime = _runtime(ctx)
    runtime.settings.data_dir.mkdir(parents=True, exist_ok=True)
    runtime.settings.index_dir.mkdir(parents=True, exist_ok=True)
    runtime.settings.review_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime.settings.project_root / "config.local.yaml"
    example = runtime.settings.project_root / "config.example.yaml"
    if not destination.exists() and example.is_file():
        shutil.copyfile(example, destination)
        typer.echo(f"已创建配置：{destination}")
    else:
        typer.echo(f"配置保持不变：{destination}")
    typer.echo(f"数据库：{runtime.settings.database_path}")


@app.command("doctor")
def doctor(ctx: typer.Context) -> None:
    """检查本地依赖和数据路径，不发送正文。"""
    runtime = _runtime(ctx)
    settings = runtime.settings
    checks: list[tuple[str, bool, str]] = []
    try:
        count = len(discover_markdown(settings.source_dir))
        checks.append(("corpus", True, f"{settings.source_dir}（{count} 篇）"))
    except Exception as exc:
        checks.append(("corpus", False, str(exc)))
    executable = settings.ollama.executable.resolve()
    checks.append(("ollama executable", executable.is_file(), str(executable)))
    client = OllamaClient(settings.ollama)
    healthy = client.healthy()
    checks.append(("ollama service", healthy, settings.ollama.host))
    if healthy:
        try:
            models = client.installed_models()
            present = any(item.split(":latest")[0] == settings.ollama.model for item in models)
            checks.append(("ollama model", present, settings.ollama.model))
        except Exception as exc:
            checks.append(("ollama model", False, type(exc).__name__))
    for name, ok, detail in checks:
        typer.echo(f"{'OK' if ok else 'FAIL'}  {name}: {detail}")
    if not all(ok for _, ok, _ in checks):
        raise typer.Exit(1)


@corpus_app.command("scan")
def corpus_scan(ctx: typer.Context) -> None:
    """只读扫描顶层 Markdown 并更新增量状态。"""
    runtime = _runtime(ctx)
    try:
        files = discover_markdown(runtime.settings.source_dir)
        typer.echo(f"语料目录：{runtime.settings.source_dir}")
        typer.echo(f"发现文章：{len(files)}")
        with runtime.database.session() as session:
            summary = scan_corpus(
                session,
                runtime.settings.source_dir,
                runtime.settings.corpus.excluded_article_ids,
            )
        typer.echo(
            f"排除 {summary.excluded}，新增 {summary.added}，修改 {summary.modified}，"
            f"元数据更新 {summary.metadata_updated}，未变化 {summary.unchanged}，"
            f"缺失 {summary.missing}，待清洗审核 {summary.needs_cleanup_review}"
        )
        if summary.unmatched_excluded_ids:
            typer.echo(f"警告：未匹配排除 ID：{', '.join(summary.unmatched_excluded_ids)}")
    except Exception as exc:
        _abort(exc)


@corpus_app.command("manage")
def corpus_manage(
    ctx: typer.Context,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="只打印地址，不自动打开浏览器")
    ] = False,
) -> None:
    """启动本地语料管理页面。"""
    runtime = _runtime(ctx)
    try:
        run_corpus_dashboard(
            runtime.database,
            runtime.settings.source_dir,
            runtime.settings.corpus.excluded_article_ids,
            runtime.settings.review_dir,
            runtime.settings.index_dir,
            _hybrid_index_builder(runtime.settings),
            open_browser=not no_open,
        )
    except Exception as exc:
        _abort(exc)


@corpus_app.command("resolve-cleanup")
def corpus_resolve_cleanup(
    ctx: typer.Context,
    article_version: str,
    keep_all: Annotated[bool, typer.Option("--keep-all", help="确认保留全部正文")] = False,
    truncate_from: Annotated[
        int | None, typer.Option("--truncate-from", min=2, help="从此单元开始删除尾部")
    ] = None,
) -> None:
    """人工裁决歧义清洗。"""
    if keep_all == (truncate_from is not None):
        _abort(ValueError("必须且只能指定 --keep-all 或 --truncate-from"))
    runtime = _runtime(ctx)
    try:
        with runtime.database.session() as session:
            resolved_id = resolve_cleanup(session, article_version, truncate_from)
        typer.echo(f"清洗裁决已保存，新版本：{resolved_id}")
    except Exception as exc:
        _abort(exc)


@chunks_app.command("propose")
def chunks_propose(
    ctx: typer.Context,
    article_ids: Annotated[list[str] | None, typer.Argument(help="可选文章 ID 列表")] = None,
) -> None:
    """使用本地 Ollama 为待处理文章生成话题块提案。"""
    runtime = _runtime(ctx)
    client = OllamaClient(runtime.settings.ollama)
    try:
        client.start_server()
        models = client.installed_models()
        if not any(item.split(":latest")[0] == runtime.settings.ollama.model for item in models):
            raise RuntimeError(f"模型未安装：{runtime.settings.ollama.model}；请先执行 ollama pull")
        with runtime.database.session() as session:
            version_ids = _latest_pending_version_ids(session, article_ids)
            if not version_ids:
                raise ValueError("没有待分块的文章版本")
        failed = False
        for version_id in version_ids:
            with runtime.database.session() as session:
                version = session.get(ArticleVersionRow, version_id)
                if version is None:
                    raise RuntimeError(f"待分块版本已消失：{version_id}")
                outcome = propose_chunks(
                    session,
                    version,
                    client,
                    runtime.settings.ollama.model,
                    runtime.settings.ollama.max_retries,
                    provider_id="ollama",
                )
                typer.echo(f"提案 {outcome.proposal_id}：{version.id}")
                for warning in outcome.warnings:
                    typer.echo(f"  警告：{warning}")
                failed = failed or outcome.status == ProposalStatus.FAILED
        if failed:
            raise RuntimeError("部分提案失败；请查看 chunks list 后重试")
    except Exception as exc:
        _abort(exc)


@chunks_app.command("retry")
def chunks_retry(
    ctx: typer.Context,
    article_id: Annotated[str, typer.Argument(help="单篇文章 ID")],
    provider: Annotated[
        str, typer.Option("--provider", help="ollama、codex-cli 或外部 chunking profile")
    ] = "ollama",
    model: Annotated[str | None, typer.Option("--model", help="覆盖 profile 默认模型")] = None,
    prompt_file: Annotated[
        Path | None, typer.Option("--prompt-file", help="仅本次使用的完整分块指令")
    ] = None,
) -> None:
    """为单篇文章生成不覆盖历史的并列分块提案。"""
    runtime = _runtime(ctx)
    try:
        custom_prompt: str | None = None
        if prompt_file is not None:
            custom_prompt = prompt_file.resolve().read_text(encoding="utf-8").strip()
            if not custom_prompt:
                raise ValueError("自定义 Prompt 不能为空")

        with runtime.database.session() as session:
            version_ids = _latest_pending_version_ids(session, [article_id])
            if not version_ids:
                raise ValueError("指定文章没有可重试的 active 待分块版本")
            version_id = version_ids[0]
            version = session.get(ArticleVersionRow, version_id)
            article = session.get(ArticleRow, version.article_id) if version is not None else None
            if version is None or article is None:
                raise ValueError("文章或文章版本不存在")
            article_title = article.title
            character_count = len(version.canonical_text)
            unit_count = session.scalar(
                select(func.count())
                .select_from(ParagraphRow)
                .where(ParagraphRow.version_id == version_id)
            )

        if provider == "ollama":
            selected_model = model or runtime.settings.ollama.model
            ollama_settings = runtime.settings.ollama.model_copy(update={"model": selected_model})
            generator = OllamaClient(ollama_settings)
            generator.start_server()
            models = generator.installed_models()
            if not any(item.split(":latest")[0] == selected_model for item in models):
                raise RuntimeError(f"模型未安装：{selected_model}；请先执行 ollama pull")
            max_retries = ollama_settings.max_retries
        elif provider == "codex-cli":
            codex_settings = runtime.settings.chunking.codex_cli
            selected_model = model or codex_settings.default_model
            typer.echo(f"文章：{article_title}")
            typer.echo(f"文章版本：{version_id}")
            typer.echo(f"清洗后正文：{character_count} 字符，{unit_count or 0} 个原子单元")
            typer.echo("外部服务：OpenAI Codex CLI（复用当前 ChatGPT 登录）")
            typer.echo(f"模型：{selected_model}")
            typer.echo("训练使用取决于当前 ChatGPT 账户的数据控制设置。")
            typer.echo(f"本次操作最多发送 {codex_settings.max_retries + 1} 次完整清洗后正文。")
            if not typer.confirm("确认将这篇文章发送给 OpenAI Codex？"):
                typer.echo("已取消，未发送正文，也未创建提案。")
                return
            codex_client = CodexCliClient(codex_settings, selected_model)
            codex_client.preflight()
            generator = codex_client
            max_retries = codex_settings.max_retries
        else:
            profile = runtime.settings.chunking.external_profiles.get(provider)
            if profile is None:
                raise ValueError(f"外部分块 profile 不存在：{provider}")
            selected_model = model or profile.default_model
            typer.echo(f"文章：{article_title}")
            typer.echo(f"文章版本：{version_id}")
            typer.echo(f"清洗后正文：{character_count} 字符，{unit_count or 0} 个原子单元")
            typer.echo(f"外部服务：{provider}  {profile.base_url}")
            typer.echo(f"模型：{selected_model}")
            typer.echo(f"本次操作最多发送 {profile.max_retries + 1} 次完整清洗后正文。")
            if not typer.confirm("确认将这篇文章发送给外部模型？"):
                typer.echo("已取消，未发送正文，也未创建提案。")
                return
            api_key = os.environ.get(profile.api_key_env)
            if not api_key:
                raise ValueError(f"环境变量未设置：{profile.api_key_env}")
            generator = OpenAICompatibleClient(profile, selected_model, api_key)
            max_retries = profile.max_retries

        with runtime.database.session() as session:
            version = session.get(ArticleVersionRow, version_id)
            if version is None:
                raise RuntimeError(f"待分块版本已消失：{version_id}")
            outcome = propose_chunks(
                session,
                version,
                generator,
                selected_model,
                max_retries,
                provider_id=provider,
                custom_prompt=custom_prompt,
                retry_id=uuid4().hex,
            )
        typer.echo(f"并列提案 {outcome.proposal_id}：{outcome.version_id}")
        for warning in outcome.warnings:
            typer.echo(f"  警告：{warning}")
        if outcome.status == ProposalStatus.FAILED:
            raise RuntimeError("提案失败；请查看 chunks list 后重试")
    except Exception as exc:
        _abort(exc)


def _latest_pending_version_ids(
    session: Session, article_ids: list[str] | None = None
) -> list[str]:
    statement = select(ArticleVersionRow).where(ArticleVersionRow.active.is_(True))
    if article_ids:
        statement = statement.where(ArticleVersionRow.article_id.in_(article_ids))
    active_versions = session.scalars(
        statement.order_by(
            ArticleVersionRow.article_id,
            ArticleVersionRow.created_at.desc(),
            ArticleVersionRow.id.desc(),
        )
    ).all()
    latest_by_article: dict[str, ArticleVersionRow] = {}
    for version in active_versions:
        latest_by_article.setdefault(version.article_id, version)
    pending = [
        version
        for version in latest_by_article.values()
        if version.status
        in {
            VersionStatus.PENDING_CHUNKING.value,
            VersionStatus.PENDING_REVIEW.value,
        }
    ]
    return [version.id for version in sorted(pending, key=lambda item: item.created_at)]


@chunks_app.command("list")
def chunks_list(ctx: typer.Context) -> None:
    """列出清洗、分块和审核状态。"""
    runtime = _runtime(ctx)
    with runtime.database.session() as session:
        rows = session.execute(
            select(ArticleVersionRow, ArticleRow)
            .join(ArticleRow, ArticleVersionRow.article_id == ArticleRow.id)
            .where(ArticleVersionRow.active.is_(True))
            .order_by(ArticleVersionRow.created_at.desc())
        ).all()
        for version, article in rows:
            proposal_count = session.scalar(
                select(func.count())
                .select_from(ChunkProposalRow)
                .where(ChunkProposalRow.version_id == version.id)
            )
            typer.echo(
                f"{version.id}  {version.status:24}  proposals={proposal_count}  {article.title}"
            )
            proposals = session.scalars(
                select(ChunkProposalRow)
                .where(ChunkProposalRow.version_id == version.id)
                .order_by(ChunkProposalRow.created_at.desc())
            ).all()
            for proposal in proposals:
                typer.echo(
                    f"  {proposal.id}  {proposal.status:16}  "
                    f"{proposal.provider_id}/{proposal.model_id}  {proposal.prompt_version}"
                )


@chunks_app.command("export-review")
def chunks_export_review(
    ctx: typer.Context,
    article_version: str,
    proposal: Annotated[str | None, typer.Option("--proposal", help="待审核提案 ID")] = None,
) -> None:
    """导出机器可读 JSON 与人工阅读 Markdown。"""
    runtime = _runtime(ctx)
    try:
        with runtime.database.session() as session:
            json_path, md_path = export_review(
                session, article_version, runtime.settings.review_dir, proposal
            )
        typer.echo(str(json_path))
        typer.echo(str(md_path))
    except Exception as exc:
        _abort(exc)


@chunks_app.command("review")
def chunks_review(
    ctx: typer.Context,
    article_version: Annotated[str, typer.Argument(help="文章版本 ID")],
    proposal: Annotated[str, typer.Option("--proposal", help="待审核提案 ID")],
    no_open: Annotated[
        bool, typer.Option("--no-open", help="只打印本地地址，不自动打开浏览器")
    ] = False,
) -> None:
    """启动只绑定本机回环地址的分块审核界面。"""
    runtime = _runtime(ctx)
    try:
        run_review_server(
            runtime.database,
            article_version,
            proposal,
            runtime.settings.review_dir,
            runtime.settings.index_dir,
            _hybrid_index_builder(runtime.settings),
            open_browser=not no_open,
        )
    except Exception as exc:
        _abort(exc)


@chunks_app.command("approve")
def chunks_approve(ctx: typer.Context, manifest: Path) -> None:
    """验证并批准人工审核后的 manifest。"""
    runtime = _runtime(ctx)
    try:
        with runtime.database.session() as session:
            manifest_id = approve_manifest(session, manifest.resolve())
        typer.echo(f"已批准不可变清单：{manifest_id}")
    except Exception as exc:
        _abort(exc)


@index_app.command("build")
def index_build(
    ctx: typer.Context,
    profile: Annotated[
        str, typer.Option("--profile", help="构建 hybrid 或兼容的 bm25 Profile")
    ] = "hybrid",
) -> None:
    """在临时目录构建索引，并在自检后发布。"""
    runtime = _runtime(ctx)
    provider = None
    try:
        if profile not in {"bm25", "hybrid"}:
            raise ValueError("--profile 只支持 bm25 或 hybrid")
        if profile == "hybrid":
            provider = _embedding_provider(runtime.settings)
        with runtime.database.session() as session:
            outcome = (
                build_hybrid_index(
                    session,
                    runtime.settings.index_dir,
                    provider,
                    bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                    dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                    rrf_k=runtime.settings.retrieval.rrf_k,
                )
                if provider is not None
                else build_index(session, runtime.settings.index_dir)
            )
        typer.echo(
            f"活动索引：{outcome.build_id}，{outcome.chunk_count} 块，"
            f"{outcome.duration_ms} ms，{outcome.path}"
        )
    except IndexBuildError as exc:
        with runtime.database.session() as session:
            record_build_failure(session, exc)
        _abort(exc)
    except Exception as exc:
        _abort(exc)
    finally:
        if provider is not None:
            provider.close()


@app.command("search")
def search_command(
    ctx: typer.Context,
    query: str,
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 5,
    mode: Annotated[RetrievalMode | None, typer.Option("--mode")] = None,
    build: Annotated[str | None, typer.Option("--build")] = None,
    rerank: Annotated[bool, typer.Option("--rerank", help="启用本地 Cross-Encoder 重排")] = False,
    rerank_candidate_k: Annotated[
        int | None,
        typer.Option("--rerank-candidate-k", min=1, max=1000, help="覆盖重排候选深度"),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="输出完整结构化候选")] = False,
    assemble_output: Annotated[
        bool,
        typer.Option("--assemble", help="将排序候选组装为本地证据上下文；不调用 Generator"),
    ] = False,
) -> None:
    """搜索已批准话题块并展示原文引用。"""
    runtime = _runtime(ctx)
    provider = None
    reranker = None
    try:
        if rerank_candidate_k is not None and not rerank:
            raise ValueError("--rerank-candidate-k 只能与 --rerank 一起使用")
        selected_mode = _resolved_retrieval_mode(runtime, mode, build)
        if rerank and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rerank 第一版只支持 --mode hybrid")
        if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
            provider = _embedding_provider(runtime.settings)
        if rerank:
            reranker = _reranker_provider(runtime.settings)
        with runtime.database.session() as session:
            candidates = search(
                session,
                query,
                top_k,
                build_id=build,
                mode=selected_mode,
                embedding_provider=provider,
                bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                rrf_k=runtime.settings.retrieval.rrf_k,
                reranker=reranker,
                rerank_candidate_k=(
                    rerank_candidate_k or runtime.settings.reranker.candidate_k
                ),
            )
        if assemble_output:
            result = assemble_context(candidates, _assembly_config(runtime.settings))
            if as_json:
                typer.echo(
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
                )
            else:
                _echo_assembly(result)
            return
        if as_json:
            typer.echo(
                json.dumps([item.model_dump() for item in candidates], ensure_ascii=False, indent=2)
            )
            return
        for rank, item in enumerate(candidates, start=1):
            snippet = item.original_text[:240].replace("\n", " ")
            score = item.retrieval_scores.reranker
            if score is None:
                score = item.retrieval_scores.rrf
            if score is None:
                score = item.retrieval_scores.dense
            if score is None:
                score = item.retrieval_scores.bm25
            typer.echo(
                f"{rank}. [{float(score or 0.0):.4f}] {item.article_title} "
                f"{item.article_version} ¶{item.paragraph_range[0]}–{item.paragraph_range[1]}"
            )
            typer.echo(f"   原文：{snippet}")
    except Exception as exc:
        _abort(exc)
    finally:
        if reranker is not None:
            reranker.close()
        if provider is not None:
            provider.close()


@app.command("ask")
def ask_command(
    ctx: typer.Context,
    question: str,
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 5,
    mode: Annotated[RetrievalMode | None, typer.Option("--mode")] = None,
    build: Annotated[str | None, typer.Option("--build")] = None,
    rerank: Annotated[
        bool, typer.Option("--rerank", help="启用本地 Cross-Encoder 重排")
    ] = False,
    rerank_candidate_k: Annotated[
        int | None,
        typer.Option("--rerank-candidate-k", min=1, max=1000, help="覆盖重排候选深度"),
    ] = None,
    supplement: Annotated[
        SupplementPolicy,
        typer.Option("--supplement", help="模型补充分析策略：auto、on 或 off"),
    ] = SupplementPolicy.AUTO,
    as_json: Annotated[bool, typer.Option("--json", help="输出结构化回答")] = False,
) -> None:
    """检索本地证据，并显式调用外部 DeepSeek 生成带引用的单轮回答。"""
    runtime = _runtime(ctx)
    provider = None
    reranker = None
    try:
        if rerank_candidate_k is not None and not rerank:
            raise ValueError("--rerank-candidate-k 只能与 --rerank 一起使用")
        selected_mode = _resolved_retrieval_mode(runtime, mode, build)
        if rerank and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rerank 第一版只支持 --mode hybrid")
        if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
            provider = _embedding_provider(runtime.settings)
        if rerank:
            reranker = _reranker_provider(runtime.settings)
        with runtime.database.session() as session:
            candidates = search(
                session,
                question,
                top_k,
                build_id=build,
                mode=selected_mode,
                embedding_provider=provider,
                bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                rrf_k=runtime.settings.retrieval.rrf_k,
                reranker=reranker,
                rerank_candidate_k=(
                    rerank_candidate_k or runtime.settings.reranker.candidate_k
                ),
            )
        context = assemble_context(candidates, _assembly_config(runtime.settings))
        if context.status == AssemblyStatus.NO_EVIDENCE:
            result = AnswerGenerationResult(
                answer=GeneratedAnswer(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    limitations=["本次检索未找到可用于回答的证据。"],
                ),
                request_count=0,
            )
        else:
            settings = runtime.settings.generator
            if settings.provider != "deepseek":
                raise ValueError(f"不支持的 Generator provider：{settings.provider}")
            api_key = os.environ.get(settings.api_key_env)
            if not api_key:
                raise ValueError(f"环境变量未设置：{settings.api_key_env}")
            typer.echo("即将把最终入选的少量证据发送给外部 Generator：", err=True)
            typer.echo(f"  服务：DeepSeek  {settings.base_url}", err=True)
            typer.echo(f"  模型：{settings.model}", err=True)
            typer.echo(
                f"  证据：{context.usage.selected_count} 块，"
                f"原文 {context.usage.original_text_chars} 字符",
                err=True,
            )
            typer.echo("  最大外部请求次数：2", err=True)
            generator = EvidenceBoundGenerator(
                DeepSeekClient(settings, api_key),
                provider_id="deepseek",
                model_id=settings.model,
            )
            result = generator.generate(question, context, supplement)
        if as_json:
            typer.echo(_answer_json(result, context))
        else:
            typer.echo(render_answer(result.answer, context))
    except Exception as exc:
        _abort(exc)
    finally:
        if reranker is not None:
            reranker.close()
        if provider is not None:
            provider.close()


@generator_app.command("models")
def generator_models(ctx: typer.Context) -> None:
    """查询当前 DeepSeek 账号可用的模型 ID；不发送问题或语料。"""
    runtime = _runtime(ctx)
    try:
        settings = runtime.settings.generator
        if settings.provider != "deepseek":
            raise ValueError(f"不支持的 Generator provider：{settings.provider}")
        api_key = os.environ.get(settings.api_key_env)
        if not api_key:
            raise ValueError(f"环境变量未设置：{settings.api_key_env}")
        for model_id in DeepSeekClient(settings, api_key).list_models():
            typer.echo(model_id)
    except Exception as exc:
        _abort(exc)


@eval_app.command("run")
def eval_run(
    ctx: typer.Context,
    evalset: str,
    k: Annotated[int, typer.Option("--k", min=1, max=100)] = 5,
    build: Annotated[
        str | None, typer.Option("--build", help="指定兼容冻结快照的 Index Build")
    ] = None,
    mode: Annotated[RetrievalMode | None, typer.Option("--mode")] = None,
    rerank: Annotated[bool, typer.Option("--rerank", help="启用本地 Cross-Encoder 重排")] = False,
    rerank_candidate_k: Annotated[
        int | None,
        typer.Option("--rerank-candidate-k", min=1, max=1000, help="覆盖重排候选深度"),
    ] = None,
) -> None:
    """运行冻结检索评测集。"""
    runtime = _runtime(ctx)
    provider = None
    reranker = None
    try:
        if rerank_candidate_k is not None and not rerank:
            raise ValueError("--rerank-candidate-k 只能与 --rerank 一起使用")
        legacy_path = Path(evalset)
        selected_mode = (
            RetrievalMode.BM25
            if legacy_path.is_file()
            else _resolved_retrieval_mode(runtime, mode, build)
        )
        if rerank and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rerank 第一版只支持 --mode hybrid 的冻结评测")
        if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
            provider = _embedding_provider(runtime.settings)
        if rerank:
            reranker = _reranker_provider(runtime.settings)
        with runtime.database.session() as session:
            outcome = (
                run_evaluation(session, legacy_path.resolve(), k)
                if legacy_path.is_file()
                else run_frozen_evaluation(
                    session,
                    evalset,
                    k,
                    build_id=build,
                    mode=selected_mode,
                    embedding_provider=provider,
                    bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                    dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                    rrf_k=runtime.settings.retrieval.rrf_k,
                    reranker=reranker,
                    rerank_candidate_k=(
                        rerank_candidate_k or runtime.settings.reranker.candidate_k
                    ),
                )
            )
        typer.echo(json.dumps(outcome, ensure_ascii=False, indent=2))
    except Exception as exc:
        _abort(exc)
    finally:
        if reranker is not None:
            reranker.close()
        if provider is not None:
            provider.close()


@eval_app.command("curate")
def eval_curate(
    ctx: typer.Context,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="只打印地址，不自动打开浏览器")
    ] = False,
) -> None:
    """启动本地评测集策展与证据审核工作台。"""
    runtime = _runtime(ctx)
    try:
        run_evaluation_server(
            runtime.database,
            runtime.settings.chunking.codex_cli,
            open_browser=not no_open,
        )
    except Exception as exc:
        _abort(exc)


if __name__ == "__main__":
    app()
