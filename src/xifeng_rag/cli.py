from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xifeng_rag import __version__
from xifeng_rag.article_evaluation import (
    compare_article_discovery_evaluations,
    load_article_discovery_dataset,
    load_article_discovery_evaluation,
    run_article_discovery_evaluation,
    save_article_discovery_evaluation,
)
from xifeng_rag.article_overviews import (
    approve_article_overview,
    propose_article_overview,
    revise_pending_article_overview,
)
from xifeng_rag.article_search import (
    ArticleIndexBuildError,
    build_article_index,
    record_article_build_failure,
    search_articles,
)
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
from xifeng_rag.corpus import (
    CLEANER_VERSION,
    analyze_trailing_catalog,
    clean_markdown,
    discover_markdown,
    extract_title,
    scan_corpus,
    sha256_text,
)
from xifeng_rag.corpus_ui import run_corpus_dashboard
from xifeng_rag.domain import (
    AssemblyResult,
    AssemblyStatus,
    BuildStatus,
    ContextAssemblyConfigSnapshot,
    EvidenceAssessment,
    EvidenceAssessmentStatus,
    LayeredRetrievalMode,
    OverviewStatus,
    ProposalStatus,
    RetrievalCandidate,
    RetrievalMode,
    VersionStatus,
)
from xifeng_rag.evaluation import run_evaluation, run_frozen_evaluation
from xifeng_rag.evaluation_ui import run_evaluation_server
from xifeng_rag.evidence_assessment import assess_evidence
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
    build_layered_index,
    record_build_failure,
    search,
    search_layered,
    search_layered_with_query_variants,
    search_with_query_variants,
)
from xifeng_rag.layered_article_evaluation import (
    FallbackPolicy,
    compare_layered_article_evaluations,
    load_layered_article_evaluation,
    run_layered_article_evaluation,
    save_layered_article_evaluation,
    simulate_layered_fallback,
)
from xifeng_rag.layered_article_search import (
    LATEST_SUITE_FILE,
    ArticleSearchStrategy,
    build_layered_article_indexes,
    load_layered_suite,
    search_articles_with_disagreement_cascade,
)
from xifeng_rag.query_rewrite import (
    NoopQueryRewriter,
    OllamaQueryRewriter,
    QueryRewriter,
    QueryRewriteResult,
    QueryRewriteStatus,
    build_query_variants,
)
from xifeng_rag.reranking import SentenceTransformersReranker
from xifeng_rag.review_ui import ArticleIndexBuilder, IndexBuilder, run_review_server
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleOverviewRow,
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
overviews_app = typer.Typer(help="文章整体概括生成与审核")
articles_app = typer.Typer(help="独立的文章发现与主题反查")
app.add_typer(corpus_app, name="corpus")
app.add_typer(chunks_app, name="chunks")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(generator_app, name="generator")
app.add_typer(overviews_app, name="overviews")
app.add_typer(articles_app, name="articles")


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


def _query_rewriter_provider(settings: Settings) -> QueryRewriter:
    selected = settings.query_rewrite.provider
    if selected == "ollama":
        return OllamaQueryRewriter(
            OllamaClient(settings.ollama),
            settings.ollama.model,
            max_retries=settings.ollama.max_retries,
        )
    if selected == "noop":
        return NoopQueryRewriter()
    raise ValueError(f"不支持的 Query Rewrite provider：{selected}")


def _echo_query_rewrite(result: QueryRewriteResult) -> None:
    if result.status in {QueryRewriteStatus.SUCCESS, QueryRewriteStatus.PARTIAL}:
        typer.echo(
            f"查询改写：{result.provider}/{result.model_id or '—'}，"
            f"{len(result.variants)} 条，{result.latency_ms:.0f} ms",
            err=True,
        )
        for variant in result.variants:
            typer.echo(f"  - {variant}", err=True)
        return
    if result.status == QueryRewriteStatus.FAILED:
        typer.echo(
            f"查询改写失败（{result.diagnostic or 'unknown'}），已退回原问题检索。",
            err=True,
        )


def _echo_evidence_assessment(result: EvidenceAssessment) -> None:
    if result.status == EvidenceAssessmentStatus.FAILED:
        typer.echo(f"证据观测失败：{result.diagnostic or 'unknown'}", err=True)
        return

    def render(value: float | bool | None) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "是" if value else "否"
        return f"{value:.4f}"

    typer.echo(
        "证据观测："
        f"候选={result.candidate_count}，文章={result.article_count_at_k}，"
        f"Top1={render(result.top1_reranker_score)}，"
        f"Δ1-2={render(result.top1_top2_margin)}，"
        f"双路率={render(result.dual_retrieval_rate_at_k)}，"
        f"原问/改写共有率={render(result.shared_by_original_and_rewrite_rate_at_k)}",
        err=True,
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


def _article_index_builder(settings: Settings) -> ArticleIndexBuilder:
    def builder(session: Session, index_dir: Path):
        provider = _embedding_provider(settings)
        try:
            return build_article_index(
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
        if build is not None
        and (
            (Path(build.path) / "qdrant").is_dir()
            or (Path(build.path) / "child" / "qdrant").is_dir()
        )
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
    return ContextAssemblyConfigSnapshot.model_validate(settings.context_assembly.model_dump())


def _search_with_profile(
    session: Session,
    runtime: Runtime,
    query: str,
    top_k: int,
    *,
    build: str | None,
    selected_mode: RetrievalMode,
    provider: SentenceTransformersEmbeddingProvider | None,
    reranker: SentenceTransformersReranker | None,
    rerank_candidate_k: int,
    layered: bool,
    rewrite_result: QueryRewriteResult | None = None,
) -> list[RetrievalCandidate]:
    query_variants = None
    if rewrite_result is not None and rewrite_result.variants:
        rewrite_settings = runtime.settings.query_rewrite
        query_variants = build_query_variants(
            query,
            rewrite_result.variants,
            original_weight=rewrite_settings.original_weight,
            rewrite_total_weight=rewrite_settings.rewrite_total_weight,
        )
    if layered:
        if build is None:
            raise ValueError("--layered 必须同时用 --build 指定分层实验索引")
        if selected_mode != RetrievalMode.HYBRID or provider is None or reranker is None:
            raise ValueError("--layered 只支持显式启用的 Hybrid + Reranker")
        settings = runtime.settings.layered
        if query_variants is not None:
            rewrite_settings = runtime.settings.query_rewrite
            return search_layered_with_query_variants(
                session,
                query,
                query_variants,
                build_id=build,
                embedding_provider=provider,
                reranker=reranker,
                top_k=top_k,
                parent_candidate_k=settings.parent_candidate_k,
                child_candidate_k=settings.child_candidate_k,
                candidate_limit=settings.candidate_limit,
                rrf_k=runtime.settings.retrieval.rrf_k,
                neighbor_window=settings.neighbor_window,
                query_fusion_rrf_k=rewrite_settings.fusion_rrf_k,
                query_candidate_limit=rewrite_settings.candidate_limit,
            )
        return search_layered(
            session,
            query,
            build_id=build,
            embedding_provider=provider,
            reranker=reranker,
            top_k=top_k,
            parent_candidate_k=settings.parent_candidate_k,
            child_candidate_k=settings.child_candidate_k,
            candidate_limit=settings.candidate_limit,
            rrf_k=runtime.settings.retrieval.rrf_k,
            neighbor_window=settings.neighbor_window,
        )
    if query_variants is not None:
        if selected_mode != RetrievalMode.HYBRID or provider is None or reranker is None:
            raise ValueError("--rewrite 只支持显式启用的 Hybrid + Reranker")
        rewrite_settings = runtime.settings.query_rewrite
        return search_with_query_variants(
            session,
            query,
            query_variants,
            top_k=top_k,
            build_id=build,
            mode=selected_mode,
            embedding_provider=provider,
            bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
            dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
            rrf_k=runtime.settings.retrieval.rrf_k,
            reranker=reranker,
            rerank_candidate_k=rerank_candidate_k,
            query_fusion_rrf_k=rewrite_settings.fusion_rrf_k,
            query_candidate_limit=rewrite_settings.candidate_limit,
        )
    return search(
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
        rerank_candidate_k=rerank_candidate_k,
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


def _answer_json(
    result: AnswerGenerationResult,
    context: AssemblyResult,
    assessment: EvidenceAssessment | None = None,
) -> str:
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
    payload = {
        **result.model_dump(mode="json"),
        "rendered_answer": render_answer(result.answer, context),
        "sources": sources,
    }
    if assessment is not None:
        payload["evidence_assessment"] = assessment.model_dump(mode="json")
    return json.dumps(
        payload,
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
def corpus_scan(
    ctx: typer.Context,
    include_modified: Annotated[
        bool,
        typer.Option(
            "--include-modified",
            help="同时导入正文已变化的既有文章；默认只报告并跳过",
        ),
    ] = False,
) -> None:
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
                include_modified=include_modified,
            )
        typer.echo(
            f"排除 {summary.excluded}，新增 {summary.added}，"
            f"已导入正文更新 {summary.modified}，"
            f"已跳过正文更新 {summary.skipped_modified}，"
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
        codex_settings = runtime.settings.chunking.codex_cli

        def subtopic_generator(model: str) -> CodexCliClient:
            client = CodexCliClient(codex_settings, model)
            client.preflight()
            return client

        run_corpus_dashboard(
            runtime.database,
            runtime.settings.source_dir,
            runtime.settings.corpus.excluded_article_ids,
            runtime.settings.review_dir,
            runtime.settings.index_dir,
            _hybrid_index_builder(runtime.settings),
            article_index_dir=runtime.settings.article_index_dir,
            article_index_builder=_article_index_builder(runtime.settings),
            subtopic_generator_factory=subtopic_generator,
            subtopic_model=codex_settings.default_model,
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


@overviews_app.command("propose")
def overviews_propose(
    ctx: typer.Context,
    article_ids: Annotated[list[str] | None, typer.Argument(help="可选文章 ID 列表")] = None,
) -> None:
    """仅根据已批准的完整分块元数据，使用本地 Ollama 生成文章概括草稿。"""
    runtime = _runtime(ctx)
    client = OllamaClient(runtime.settings.ollama)
    process = None
    try:
        with runtime.database.session() as session:
            statement = (
                select(ArticleVersionRow)
                .join(
                    ApprovedManifestRow,
                    ApprovedManifestRow.version_id == ArticleVersionRow.id,
                )
                .where(ArticleVersionRow.active.is_(True))
            )
            if article_ids:
                statement = statement.where(ArticleVersionRow.article_id.in_(article_ids))
            versions = list(
                session.scalars(
                    statement.order_by(
                        ArticleVersionRow.article_id,
                        ArticleVersionRow.created_at.desc(),
                    )
                ).all()
            )
            selected: dict[str, str] = {}
            for version in versions:
                selected.setdefault(version.article_id, version.id)
            version_ids = list(selected.values())
        if not version_ids:
            raise ValueError("没有带批准分块清单的活动文章版本")
        process = client.start_server()
        models = client.installed_models()
        if not any(item.split(":latest")[0] == runtime.settings.ollama.model for item in models):
            raise RuntimeError(f"模型未安装：{runtime.settings.ollama.model}；请先执行 ollama pull")
        failed = False
        for version_id in version_ids:
            try:
                with runtime.database.session() as session:
                    row = propose_article_overview(
                        session,
                        version_id,
                        client,
                        runtime.settings.ollama.model,
                        provider_id="ollama",
                    )
                    version = session.get(ArticleVersionRow, version_id)
                    if version is None:
                        raise RuntimeError(f"文章版本已消失：{version_id}")
                    article = session.get(ArticleRow, version.article_id)
                    title = article.title if article is not None else version_id
                typer.echo(f"{row.id}  {row.status}  {title}")
                failed = failed or row.status == "failed"
            except Exception as exc:
                failed = True
                typer.echo(f"{version_id}  failed  {type(exc).__name__}: {exc}", err=True)
        if failed:
            raise RuntimeError("部分文章概括生成失败")
    except Exception as exc:
        _abort(exc)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


@overviews_app.command("list")
def overviews_list(ctx: typer.Context) -> None:
    """列出文章概括草稿与批准状态。"""
    runtime = _runtime(ctx)
    with runtime.database.session() as session:
        rows = session.execute(
            select(ArticleOverviewRow, ArticleVersionRow, ArticleRow)
            .join(ArticleVersionRow, ArticleOverviewRow.version_id == ArticleVersionRow.id)
            .join(ArticleRow, ArticleVersionRow.article_id == ArticleRow.id)
            .order_by(ArticleOverviewRow.created_at.desc())
        ).all()
        if not rows:
            typer.echo("尚无文章概括")
            return
        for overview, version, article in rows:
            typer.echo(
                f"{overview.id}  {overview.status:15}  {article.title}\n"
                f"  {version.id}  {overview.provider_id}/{overview.model_id}\n"
                f"  {overview.article_summary}"
            )


@overviews_app.command("revise-pending")
def overviews_revise_pending(
    ctx: typer.Context,
    overview_ids: Annotated[list[str] | None, typer.Argument(help="可选的待审概括 ID 列表")] = None,
    provider: Annotated[str, typer.Option(help="显式使用的外部结构化生成 profile")] = "deepseek",
    model: Annotated[str | None, typer.Option(help="覆盖 profile 默认模型")] = None,
) -> None:
    """用外部模型精简待审概括；只发送标题和批准分块元数据，不发送正文。"""
    runtime = _runtime(ctx)
    try:
        profile = runtime.settings.chunking.external_profiles.get(provider)
        if profile is None:
            raise ValueError(f"外部 profile 不存在：{provider}")
        selected_model = model or profile.default_model
        api_key = os.environ.get(profile.api_key_env)
        if not api_key:
            raise ValueError(f"环境变量未设置：{profile.api_key_env}")
        with runtime.database.session() as session:
            statement = select(ArticleOverviewRow).where(
                ArticleOverviewRow.status == OverviewStatus.PENDING_REVIEW.value
            )
            if overview_ids:
                statement = statement.where(ArticleOverviewRow.id.in_(overview_ids))
            rows = list(session.scalars(statement.order_by(ArticleOverviewRow.created_at)).all())
            pending_ids = [row.id for row in rows]
        if not pending_ids:
            raise ValueError("没有符合条件的待审文章概括")
        typer.echo(
            f"将使用 {provider}/{selected_model} 精简 {len(pending_ids)} 篇待审概括；"
            "仅发送文章标题与完整批准分块元数据，不发送原文。"
        )
        generator = OpenAICompatibleClient(profile, selected_model, api_key)
        failed = False
        for overview_id in pending_ids:
            try:
                with runtime.database.session() as session:
                    replacement = revise_pending_article_overview(
                        session,
                        overview_id,
                        generator,
                        selected_model,
                        provider_id=provider,
                    )
                    version = session.get(ArticleVersionRow, replacement.version_id)
                    article = (
                        session.get(ArticleRow, version.article_id) if version is not None else None
                    )
                    title = article.title if article is not None else replacement.version_id
                typer.echo(f"{replacement.id}  pending_review  {title}")
            except Exception as exc:
                failed = True
                typer.echo(f"{overview_id}  failed  {type(exc).__name__}: {exc}", err=True)
        if failed:
            raise RuntimeError("部分待审概括精简失败；成功项已保留")
    except Exception as exc:
        _abort(exc)


@corpus_app.command("cleanup-dry-run")
def corpus_cleanup_dry_run(
    ctx: typer.Context,
    show_all: Annotated[bool, typer.Option("--all", help="同时显示清洗判断未变化的文章")] = False,
) -> None:
    """只读比较新清洗规则，不修改语料或业务数据。"""
    runtime = _runtime(ctx)
    try:
        files = discover_markdown(runtime.settings.source_dir)
        with runtime.database.session() as session:
            article_ids_by_path: dict[str, str] = {
                source_path: article_id
                for source_path, article_id in session.execute(
                    select(ArticleRow.source_path, ArticleRow.id)
                ).all()
            }
            rows = session.execute(
                select(ArticleRow.source_path, ArticleVersionRow)
                .join(ArticleVersionRow, ArticleVersionRow.article_id == ArticleRow.id)
                .where(ArticleVersionRow.active.is_(True))
                .order_by(ArticleVersionRow.created_at.desc())
            ).all()
        current_by_path: dict[str, ArticleVersionRow] = {}
        for source_path, version in rows:
            current_by_path.setdefault(source_path, version)

        excluded_ids = frozenset(runtime.settings.corpus.excluded_article_ids)
        changed = review = removable = processed = excluded = 0
        for path in files:
            relative = path.relative_to(runtime.settings.source_dir).as_posix()
            if article_ids_by_path.get(relative) in excluded_ids:
                excluded += 1
                continue
            processed += 1
            markdown = path.read_text(encoding="utf-8-sig")
            analysis = analyze_trailing_catalog(markdown)
            canonical_hash = sha256_text(clean_markdown(markdown).canonical_text)
            current = current_by_path.get(relative)
            old_start: int | None = None
            if current is not None:
                old_ranges = json.loads(current.removed_ranges_json)
                if (
                    isinstance(old_ranges, list)
                    and old_ranges
                    and isinstance(old_ranges[0], list)
                    and old_ranges[0]
                    and isinstance(old_ranges[0][0], int)
                ):
                    old_start = old_ranges[0][0]
            content_changed = current is None or current.canonical_hash != canonical_hash
            differs = (
                current is None
                or current.cleaner_version != CLEANER_VERSION
                or content_changed
                or old_start != analysis.source_start
                or (
                    analysis.status == "review"
                    and current.status != VersionStatus.NEEDS_CLEANUP_REVIEW.value
                )
            )
            changed += int(differs)
            review += int(analysis.status == "review")
            removable += int(analysis.status == "remove")
            if not differs and not show_all:
                continue
            removed_chars = (
                analysis.source_end - analysis.source_start
                if analysis.source_start is not None and analysis.source_end is not None
                else 0
            )
            title = extract_title(markdown, path.stem)
            typer.echo(
                f"{analysis.status.upper():6}  {title}  "
                f"旧起点={old_start if old_start is not None else '-'}  "
                f"新起点={analysis.source_start if analysis.source_start is not None else '-'}  "
                f"正文变化={'是' if content_changed else '否'}  "
                f"移除字符={removed_chars}  链接={analysis.link_count}  "
                f"链接行={analysis.linked_lines}"
            )
        typer.echo(
            f"共检查 {processed} 篇，排除 {excluded} 篇；判断变化 {changed}，"
            f"自动移除 {removable}，待审核 {review}。"
        )
        typer.echo("只读检查完成：未修改外部语料或业务数据。")
    except Exception as exc:
        _abort(exc)


@overviews_app.command("approve")
def overviews_approve(
    ctx: typer.Context,
    overview_id: Annotated[str, typer.Argument(help="文章概括 ID")],
    summary_file: Annotated[
        Path | None, typer.Option("--summary-file", help="可选的人工修改后纯文本文件")
    ] = None,
) -> None:
    """批准文章概括；可在批准时用本地文本文件替换模型草稿。"""
    runtime = _runtime(ctx)
    try:
        summary = None
        if summary_file is not None:
            summary = summary_file.resolve().read_text(encoding="utf-8").strip()
        with runtime.database.session() as session:
            row = approve_article_overview(session, overview_id, summary)
        typer.echo(f"已批准文章概括：{row.id}")
    except Exception as exc:
        _abort(exc)


@articles_app.command("build-index")
def articles_build_index(ctx: typer.Context) -> None:
    """只用已批准文章概括构建独立的文章级 Hybrid 索引。"""
    runtime = _runtime(ctx)
    provider = None
    try:
        provider = _embedding_provider(runtime.settings)
        with runtime.database.session() as session:
            outcome = build_article_index(
                session,
                runtime.settings.article_index_dir,
                provider,
                bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                rrf_k=runtime.settings.retrieval.rrf_k,
            )
        typer.echo(
            f"活动文章索引：{outcome.build_id}，{outcome.article_count} 篇，"
            f"{outcome.duration_ms} ms，{outcome.path}"
        )
    except ArticleIndexBuildError as exc:
        with runtime.database.session() as session:
            record_article_build_failure(session, exc)
        _abort(exc)
    except Exception as exc:
        _abort(exc)
    finally:
        if provider is not None:
            provider.close()


@articles_app.command("search")
def articles_search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="想反查文章的主题描述")],
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 5,
    mode: Annotated[
        RetrievalMode,
        typer.Option("--mode", help="累计文档兼容策略的检索模式"),
    ] = RetrievalMode.HYBRID,
    strategy: Annotated[
        ArticleSearchStrategy,
        typer.Option("--strategy", help="文章反查策略"),
    ] = (
        ArticleSearchStrategy.DISAGREEMENT_CASCADE
    ),
    build: Annotated[str | None, typer.Option("--build")] = None,
    suite_path: Annotated[
        Path | None,
        typer.Option("--suite", help="分层索引套件；默认使用最新套件"),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="输出结构化文章列表")] = False,
) -> None:
    """搜索相关文章；结果不是原文证据，也不进入现有问答权重。"""
    runtime = _runtime(ctx)
    provider = None
    try:
        if strategy == ArticleSearchStrategy.DISAGREEMENT_CASCADE or mode in {
            RetrievalMode.DENSE,
            RetrievalMode.HYBRID,
        }:
            provider = _embedding_provider(runtime.settings)
        with runtime.database.session() as session:
            if strategy == ArticleSearchStrategy.DISAGREEMENT_CASCADE:
                if build is not None:
                    raise ValueError("--build 只适用于 cumulative_hybrid 策略")
                resolved_suite = (
                    suite_path.resolve()
                    if suite_path is not None
                    else runtime.settings.article_index_dir / LATEST_SUITE_FILE
                )
                if not resolved_suite.is_file():
                    raise FileNotFoundError(
                        "默认分层文章反查索引不存在，请运行 "
                        "`uv run xifeng articles build-layered-index --candidate-k 20`"
                    )
                assert provider is not None
                outcome = search_articles_with_disagreement_cascade(
                    session,
                    query,
                    top_k,
                    embedding_provider=provider,
                    build_ids=load_layered_suite(resolved_suite),
                    fusion_rrf_k=runtime.settings.retrieval.rrf_k,
                )
                results = outcome.results
                activated_layers = [item.value for item in outcome.activated_layers]
            else:
                if suite_path is not None:
                    raise ValueError("--suite 只适用于 disagreement_cascade 策略")
                results = search_articles(
                    session,
                    query,
                    top_k,
                    mode=mode,
                    embedding_provider=provider,
                    build_id=build,
                    bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                    dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                    rrf_k=runtime.settings.retrieval.rrf_k,
                )
                activated_layers = []
        if as_json:
            typer.echo(json.dumps([asdict(item) for item in results], ensure_ascii=False))
            return
        if not results:
            typer.echo("没有找到相关文章")
            return
        if activated_layers:
            typer.echo(
                "策略=disagreement_cascade  触发层=" + ",".join(activated_layers)
            )
        for rank, result in enumerate(results, start=1):
            published = result.published_at or "发布时间未知"
            topics = "、".join(result.matched_topics) or "—"
            scores = (
                f"RRF={result.rrf_score:.6f}"
                if result.rrf_score is not None
                else f"来源={'+'.join(result.sources)}"
            )
            typer.echo(
                f"[{rank}] {result.title}  {published}\n"
                f"    匹配话题：{topics}\n"
                f"    {result.article_summary}\n"
                f"    {scores}  article_id={result.article_id}"
            )
    except Exception as exc:
        _abort(exc)
    finally:
        if provider is not None:
            provider.close()


@articles_app.command("build-layered-index")
def articles_build_layered_index(
    ctx: typer.Context,
    candidate_k: Annotated[
        int, typer.Option("--candidate-k", min=5, max=1000, help="每层单元候选深度")
    ] = 20,
) -> None:
    """构建四个互相隔离、默认不激活的文章发现实验索引。"""
    runtime = _runtime(ctx)
    provider = None
    try:
        provider = _embedding_provider(runtime.settings)
        layered = runtime.settings.layered
        with runtime.database.session() as session:
            suite = build_layered_article_indexes(
                session,
                runtime.settings.article_index_dir,
                provider,
                candidate_k=candidate_k,
                splitter_version=layered.splitter_version,
                target_chars=layered.target_chars,
                max_chars=layered.max_chars,
                tail_merge_chars=layered.tail_merge_chars,
            )
        typer.echo(
            f"分层文章索引套件：{suite.suite_id}，快照 {suite.corpus_snapshot}"
        )
        for layer, outcome in suite.builds.items():
            typer.echo(
                f"{layer.value}: Build={outcome.build_id}  单元={outcome.unit_count}  "
                f"文章={outcome.article_count}  {outcome.duration_ms}ms"
            )
        typer.echo(f"套件清单：{suite.manifest_path}")
    except Exception as exc:
        _abort(exc)
    finally:
        if provider is not None:
            provider.close()


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
            article_index_dir=runtime.settings.article_index_dir,
            article_index_builder=_article_index_builder(runtime.settings),
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
        str, typer.Option("--profile", help="构建 hybrid、layered 或兼容的 bm25 Profile")
    ] = "hybrid",
) -> None:
    """在临时目录构建索引，并在自检后发布。"""
    runtime = _runtime(ctx)
    provider = None
    try:
        if profile not in {"bm25", "hybrid", "layered"}:
            raise ValueError("--profile 只支持 bm25、hybrid 或 layered")
        if profile in {"hybrid", "layered"}:
            provider = _embedding_provider(runtime.settings)
        with runtime.database.session() as session:
            if profile == "layered":
                assert provider is not None
                layered = runtime.settings.layered
                outcome = build_layered_index(
                    session,
                    runtime.settings.index_dir,
                    provider,
                    splitter_version=layered.splitter_version,
                    target_chars=layered.target_chars,
                    max_chars=layered.max_chars,
                    tail_merge_chars=layered.tail_merge_chars,
                    parent_candidate_k=layered.parent_candidate_k,
                    child_candidate_k=layered.child_candidate_k,
                    candidate_limit=layered.candidate_limit,
                    rrf_k=runtime.settings.retrieval.rrf_k,
                )
            elif profile == "hybrid":
                assert provider is not None
                outcome = build_hybrid_index(
                    session,
                    runtime.settings.index_dir,
                    provider,
                    bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                    dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                    rrf_k=runtime.settings.retrieval.rrf_k,
                )
            else:
                outcome = build_index(session, runtime.settings.index_dir)
        if profile == "layered":
            typer.echo(
                f"分层实验索引：{outcome.build_id}，{outcome.chunk_count} 个子块，"
                f"{outcome.duration_ms} ms，{outcome.path}"
            )
            typer.echo(
                f'查询：uv run xifeng search "问题" --mode hybrid --rerank '
                f"--layered --build {outcome.build_id}"
            )
        else:
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
    layered: Annotated[
        bool, typer.Option("--layered", help="使用显式指定的分层父子实验索引")
    ] = False,
    rewrite: Annotated[
        bool, typer.Option("--rewrite", help="使用本地模型生成保守查询改写")
    ] = False,
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
    rewrite_result = None
    assessment = None
    try:
        if rerank_candidate_k is not None and not rerank:
            raise ValueError("--rerank-candidate-k 只能与 --rerank 一起使用")
        selected_mode = _resolved_retrieval_mode(runtime, mode, build)
        if layered and not rerank:
            raise ValueError("--layered 必须同时启用 --rerank")
        if layered and build is None:
            raise ValueError("--layered 必须同时用 --build 指定分层实验索引")
        if layered and rerank_candidate_k is not None:
            raise ValueError("--layered 的候选深度由 layered 配置控制")
        if rerank and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rerank 第一版只支持 --mode hybrid")
        if rewrite and not rerank:
            raise ValueError("--rewrite 必须同时启用 --rerank")
        if rewrite and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rewrite 第一版只支持 --mode hybrid")
        if rewrite:
            rewrite_result = _query_rewriter_provider(runtime.settings).rewrite(
                query, runtime.settings.query_rewrite.max_variants
            )
            _echo_query_rewrite(rewrite_result)
        if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
            provider = _embedding_provider(runtime.settings)
        if rerank:
            reranker = _reranker_provider(runtime.settings)
        with runtime.database.session() as session:
            candidates = _search_with_profile(
                session,
                runtime,
                query,
                top_k,
                build=build,
                selected_mode=selected_mode,
                provider=provider,
                reranker=reranker,
                rerank_candidate_k=(rerank_candidate_k or runtime.settings.reranker.candidate_k),
                layered=layered,
                rewrite_result=rewrite_result,
            )
        if runtime.settings.evidence_assessment.enabled:
            assessment = assess_evidence(
                candidates, runtime.settings.evidence_assessment.observation_k
            )
            _echo_evidence_assessment(assessment)
        if assemble_output:
            result = assemble_context(candidates, _assembly_config(runtime.settings))
            if as_json:
                payload: object = result.model_dump(mode="json")
                if assessment is not None:
                    payload = {
                        "context": payload,
                        "evidence_assessment": assessment.model_dump(mode="json"),
                    }
                typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                _echo_assembly(result)
            return
        if as_json:
            payload = [item.model_dump() for item in candidates]
            if assessment is not None:
                typer.echo(
                    json.dumps(
                        {
                            "candidates": payload,
                            "evidence_assessment": assessment.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
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
    rerank: Annotated[bool, typer.Option("--rerank", help="启用本地 Cross-Encoder 重排")] = False,
    rerank_candidate_k: Annotated[
        int | None,
        typer.Option("--rerank-candidate-k", min=1, max=1000, help="覆盖重排候选深度"),
    ] = None,
    layered: Annotated[
        bool, typer.Option("--layered", help="使用显式指定的分层父子实验索引")
    ] = False,
    rewrite: Annotated[
        bool, typer.Option("--rewrite", help="使用本地模型生成保守查询改写")
    ] = False,
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
    rewrite_result = None
    assessment = None
    try:
        if rerank_candidate_k is not None and not rerank:
            raise ValueError("--rerank-candidate-k 只能与 --rerank 一起使用")
        selected_mode = _resolved_retrieval_mode(runtime, mode, build)
        if layered and not rerank:
            raise ValueError("--layered 必须同时启用 --rerank")
        if layered and build is None:
            raise ValueError("--layered 必须同时用 --build 指定分层实验索引")
        if layered and rerank_candidate_k is not None:
            raise ValueError("--layered 的候选深度由 layered 配置控制")
        if rerank and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rerank 第一版只支持 --mode hybrid")
        if rewrite and not rerank:
            raise ValueError("--rewrite 必须同时启用 --rerank")
        if rewrite and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rewrite 第一版只支持 --mode hybrid")
        if rewrite:
            rewrite_result = _query_rewriter_provider(runtime.settings).rewrite(
                question, runtime.settings.query_rewrite.max_variants
            )
            _echo_query_rewrite(rewrite_result)
        if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
            provider = _embedding_provider(runtime.settings)
        if rerank:
            reranker = _reranker_provider(runtime.settings)
        with runtime.database.session() as session:
            candidates = _search_with_profile(
                session,
                runtime,
                question,
                top_k,
                build=build,
                selected_mode=selected_mode,
                provider=provider,
                reranker=reranker,
                rerank_candidate_k=(rerank_candidate_k or runtime.settings.reranker.candidate_k),
                layered=layered,
                rewrite_result=rewrite_result,
            )
        if runtime.settings.evidence_assessment.enabled:
            assessment = assess_evidence(
                candidates, runtime.settings.evidence_assessment.observation_k
            )
            _echo_evidence_assessment(assessment)
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
            typer.echo(_answer_json(result, context, assessment))
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
        model_ids = DeepSeekClient(settings, api_key).list_models()
        typer.echo("可用模型：")
        for model_id in model_ids:
            typer.echo(f"- {model_id}")
        typer.echo()
        available = settings.model in model_ids
        state = "可用" if available else "不可用"
        typer.echo(f"当前配置：{settings.model}（{state}）")
        typer.echo("推荐：deepseek-v4-flash（当前纯文本 RAG 默认选择）")
        config_path = runtime.settings.project_root / "config.local.yaml"
        if available:
            typer.echo("无需修改模型配置。")
            typer.echo('下一步：uv run xifeng ask "你的问题" --mode hybrid --rerank')
        else:
            typer.echo(f"请在 {config_path} 中设置一个可用模型，例如：")
            typer.echo("generator:")
            typer.echo("  model: deepseek-v4-flash")
        typer.echo("简短说明：docs/guides/deepseek-generator-setup.md")
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
    layered: Annotated[
        bool, typer.Option("--layered", help="使用显式指定的分层父子实验索引")
    ] = False,
    rewrite: Annotated[
        bool, typer.Option("--rewrite", help="使用本地模型生成保守查询改写")
    ] = False,
    layered_mode: Annotated[
        LayeredRetrievalMode,
        typer.Option("--layered-mode", help="分层实验召回：child 或 parent-child"),
    ] = LayeredRetrievalMode.PARENT_CHILD,
    neighbor_window: Annotated[
        int | None,
        typer.Option("--neighbor-window", min=0, max=10, help="覆盖分层相邻子块数量"),
    ] = None,
) -> None:
    """运行冻结检索评测集。"""
    runtime = _runtime(ctx)
    provider = None
    reranker = None
    query_rewriter = None
    try:
        if rerank_candidate_k is not None and not rerank:
            raise ValueError("--rerank-candidate-k 只能与 --rerank 一起使用")
        if layered and not rerank:
            raise ValueError("--layered 必须同时启用 --rerank")
        if layered and build is None:
            raise ValueError("--layered 必须同时用 --build 指定分层实验索引")
        if layered and rerank_candidate_k is not None:
            raise ValueError("--layered 的候选深度由 layered 配置控制")
        if not layered and (
            layered_mode != LayeredRetrievalMode.PARENT_CHILD or neighbor_window is not None
        ):
            raise ValueError("--layered-mode 和 --neighbor-window 只能与 --layered 一起使用")
        legacy_path = Path(evalset)
        selected_mode = (
            RetrievalMode.BM25
            if legacy_path.is_file()
            else _resolved_retrieval_mode(runtime, mode, build)
        )
        if rerank and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rerank 第一版只支持 --mode hybrid 的冻结评测")
        if rewrite and legacy_path.is_file():
            raise ValueError("--rewrite 只支持数据库中的冻结评测集")
        if rewrite and not rerank:
            raise ValueError("--rewrite 必须同时启用 --rerank")
        if rewrite and selected_mode != RetrievalMode.HYBRID:
            raise ValueError("--rewrite 第一版只支持 --mode hybrid")
        if rewrite:
            query_rewriter = _query_rewriter_provider(runtime.settings)
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
                    layered=layered,
                    layered_parent_candidate_k=(runtime.settings.layered.parent_candidate_k),
                    layered_child_candidate_k=(runtime.settings.layered.child_candidate_k),
                    layered_candidate_limit=runtime.settings.layered.candidate_limit,
                    layered_neighbor_window=(
                        neighbor_window
                        if neighbor_window is not None
                        else runtime.settings.layered.neighbor_window
                    ),
                    layered_mode=layered_mode,
                    query_rewriter=query_rewriter,
                    rewrite_max_variants=runtime.settings.query_rewrite.max_variants,
                    rewrite_original_weight=(runtime.settings.query_rewrite.original_weight),
                    rewrite_total_weight=(runtime.settings.query_rewrite.rewrite_total_weight),
                    rewrite_fusion_rrf_k=(runtime.settings.query_rewrite.fusion_rrf_k),
                    rewrite_candidate_limit=(runtime.settings.query_rewrite.candidate_limit),
                    evidence_assessment_enabled=(runtime.settings.evidence_assessment.enabled),
                    evidence_assessment_observation_k=(
                        runtime.settings.evidence_assessment.observation_k
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


@eval_app.command("article-discovery")
def eval_article_discovery(
    ctx: typer.Context,
    dataset_path: Annotated[
        Path | None,
        typer.Argument(help="文章反查评测 JSON；默认使用 data/article_discovery_queries.json"),
    ] = None,
    build: Annotated[str | None, typer.Option("--build", help="指定文章级 Index Build")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="输出完整结构化运行结果")] = False,
) -> None:
    """在人工确认题上比较文章级 BM25、Dense 与 Hybrid。"""
    runtime = _runtime(ctx)
    provider = None
    try:
        resolved_dataset = (
            dataset_path.resolve()
            if dataset_path is not None
            else runtime.settings.data_dir / "article_discovery_queries.json"
        )
        dataset = load_article_discovery_dataset(resolved_dataset)
        provider = _embedding_provider(runtime.settings)
        with runtime.database.session() as session:
            outcome = run_article_discovery_evaluation(
                session,
                dataset,
                resolved_dataset,
                provider,
                build_id=build,
                bm25_candidate_k=runtime.settings.retrieval.bm25_candidate_k,
                dense_candidate_k=runtime.settings.retrieval.dense_candidate_k,
                rrf_k=runtime.settings.retrieval.rrf_k,
            )
        output_path = save_article_discovery_evaluation(
            outcome, runtime.settings.data_dir / "evaluation_runs"
        )
        if as_json:
            typer.echo(outcome.model_dump_json(indent=2))
            return
        typer.echo(
            f"文章反查评测完成：{outcome.run_id}，确认题 {outcome.confirmed_question_count}，"
            f"跳过 {outcome.skipped_question_count}"
        )
        for mode_result in outcome.modes:
            metrics = mode_result.metrics
            values = (
                f"Hit@1={metrics.hit_at_1:.3f}  Hit@3={metrics.hit_at_3:.3f}  "
                f"Hit@5={metrics.hit_at_5:.3f}  MRR={metrics.mrr:.3f}  "
                f"平均={metrics.latency_mean_ms:.1f}ms  P95={metrics.latency_p95_ms:.1f}ms"
                if metrics.mrr is not None
                else "没有与当前 Build 兼容的确认题"
            )
            typer.echo(
                f"{mode_result.mode.value}: {values}  不兼容={metrics.incompatible_question_count}"
            )
        typer.echo(f"结果：{output_path}")
    except Exception as exc:
        _abort(exc)
    finally:
        if provider is not None:
            provider.close()


@eval_app.command("compare-article-discovery")
def compare_article_discovery(
    baseline_path: Annotated[Path, typer.Argument(help="基线文章反查评测结果 JSON")],
    current_path: Annotated[Path, typer.Argument(help="新版文章反查评测结果 JSON")],
    as_json: Annotated[bool, typer.Option("--json", help="输出完整结构化对比结果")] = False,
) -> None:
    """比较两次参数一致的文章反查评测结果。"""
    try:
        baseline = load_article_discovery_evaluation(baseline_path)
        current = load_article_discovery_evaluation(current_path)
        comparison = compare_article_discovery_evaluations(
            baseline,
            current,
            baseline_path=baseline_path,
            current_path=current_path,
        )
        if as_json:
            typer.echo(comparison.model_dump_json(indent=2))
            return

        typer.echo(
            f"文章反查评测对比：{comparison.baseline_run_id} -> {comparison.current_run_id}"
        )
        typer.echo(
            f"Build：{comparison.baseline_build_id} -> {comparison.current_build_id}"
        )
        typer.echo(
            f"Profile：{comparison.baseline_profile_id} -> {comparison.current_profile_id}"
        )
        typer.echo(
            "Git："
            f"{comparison.baseline_code_commit or '未知'}"
            f" (dirty={comparison.baseline_worktree_dirty}) -> "
            f"{comparison.current_code_commit or '未知'}"
            f" (dirty={comparison.current_worktree_dirty})"
        )
        metric_labels = {
            "hit_at_1": "Hit@1",
            "hit_at_3": "Hit@3",
            "hit_at_5": "Hit@5",
            "mrr": "MRR",
            "latency_mean_ms": "平均延迟(ms)",
            "latency_p95_ms": "P95延迟(ms)",
        }
        status_labels = {
            "improved": "改善",
            "regressed": "退化",
            "unchanged": "不变",
            "incompatible": "不兼容",
        }
        for mode in comparison.modes:
            typer.echo(f"\n[{mode.mode.value}]")
            for field, values in mode.metrics.items():
                if values.baseline is None or values.current is None or values.delta is None:
                    rendered = "不可用"
                else:
                    rendered = (
                        f"{values.baseline:.3f} -> {values.current:.3f} "
                        f"({values.delta:+.3f})"
                    )
                typer.echo(f"  {metric_labels[field]}: {rendered}")
            typer.echo("  逐题排名：")
            for question in mode.questions:
                baseline_rank = (
                    str(question.baseline_rank)
                    if question.baseline_rank is not None
                    else "未召回"
                )
                current_rank = (
                    str(question.current_rank) if question.current_rank is not None else "未召回"
                )
                typer.echo(
                    f"    {question.question_id}: {baseline_rank} -> {current_rank} "
                    f"[{status_labels[question.status]}]"
                )
    except Exception as exc:
        _abort(exc)


@eval_app.command("layered-article-discovery")
def eval_layered_article_discovery(
    ctx: typer.Context,
    dataset_path: Annotated[
        Path | None,
        typer.Argument(help="文章反查评测 JSON；默认使用 data/article_discovery_queries.json"),
    ] = None,
    suite_path: Annotated[
        Path | None,
        typer.Option("--suite", help="分层索引套件清单；默认使用最新套件"),
    ] = None,
    candidate_k: Annotated[
        int, typer.Option("--candidate-k", min=5, max=1000, help="每层单元候选深度")
    ] = 20,
    as_json: Annotated[bool, typer.Option("--json", help="输出完整结构化运行结果")] = False,
) -> None:
    """在四个隔离层上运行 BM25、Dense 与多单元聚合实验。"""
    runtime = _runtime(ctx)
    provider = None
    try:
        resolved_dataset = (
            dataset_path.resolve()
            if dataset_path is not None
            else runtime.settings.data_dir / "article_discovery_queries.json"
        )
        resolved_suite = (
            suite_path.resolve()
            if suite_path is not None
            else runtime.settings.article_index_dir / LATEST_SUITE_FILE
        )
        dataset = load_article_discovery_dataset(resolved_dataset)
        build_ids = load_layered_suite(resolved_suite)
        provider = _embedding_provider(runtime.settings)
        with runtime.database.session() as session:
            outcome = run_layered_article_evaluation(
                session,
                dataset,
                resolved_dataset,
                provider,
                build_ids,
                candidate_k=candidate_k,
                aggregation_rrf_k=runtime.settings.retrieval.rrf_k,
            )
        output_path = save_layered_article_evaluation(
            outcome, runtime.settings.data_dir / "evaluation_runs"
        )
        if as_json:
            typer.echo(outcome.model_dump_json(indent=2))
            return
        typer.echo(
            f"分层文章反查评测完成：{outcome.run_id}，"
            f"确认题 {outcome.confirmed_question_count}"
        )
        for channel in outcome.channels:
            metrics = channel.metrics
            summary = (
                f"Hit@1={metrics.hit_at_1:.3f} Hit@3={metrics.hit_at_3:.3f} "
                f"Hit@5={metrics.hit_at_5:.3f} MRR={metrics.mrr:.3f}"
                if metrics.mrr is not None
                else "该层没有兼容题"
            )
            typer.echo(
                f"{channel.layer.value}/{channel.mode.value}/{channel.aggregation.value}: "
                f"{summary}  不可用={metrics.unavailable_question_count}"
            )
        typer.echo(f"结果：{output_path}")
    except Exception as exc:
        _abort(exc)
    finally:
        if provider is not None:
            provider.close()


@eval_app.command("simulate-article-fallback")
def eval_simulate_article_fallback(
    result_path: Annotated[Path, typer.Argument(help="分层文章反查评测结果 JSON")],
    policy: Annotated[FallbackPolicy, typer.Option("--policy")] = (
        FallbackPolicy.MARGIN_CASCADE
    ),
    margin_ratio: Annotated[
        float, typer.Option("--margin-ratio", min=0.0, max=1.0)
    ] = 0.1,
    overlap: Annotated[float, typer.Option("--overlap", min=0.0, max=1.0)] = 0.5,
    rrf_k: Annotated[int, typer.Option("--rrf-k", min=1)] = 60,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """只读取已保存结果，离线重放分层降级策略。"""
    try:
        source = load_layered_article_evaluation(result_path)
        outcome = simulate_layered_fallback(
            source,
            policy=policy,
            margin_ratio_threshold=margin_ratio,
            overlap_threshold=overlap,
            fusion_rrf_k=rrf_k,
        )
        if as_json:
            typer.echo(outcome.model_dump_json(indent=2))
            return
        typer.echo(
            f"策略={outcome.policy.value}  Hit@1={outcome.hit_at_1:.3f}  "
            f"Hit@3={outcome.hit_at_3:.3f}  Hit@5={outcome.hit_at_5:.3f}  "
            f"MRR={outcome.mrr:.3f}"
        )
        typer.echo(
            "触发比例："
            + "  ".join(f"{key}={value:.1%}" for key, value in outcome.layer_trigger_rate.items())
        )
        typer.echo(
            f"延迟估算：平均 {outcome.latency_mean_ms:.1f}ms，"
            f"最坏 {outcome.latency_max_ms:.1f}ms"
        )
        typer.echo(f"改善：{', '.join(outcome.improved_question_ids) or '无'}")
        typer.echo(f"退化：{', '.join(outcome.regressed_question_ids) or '无'}")
    except Exception as exc:
        _abort(exc)


@eval_app.command("compare-layered-article-discovery")
def compare_layered_article_discovery(
    baseline_path: Annotated[Path, typer.Argument(help="基线分层文章评测 JSON")],
    current_path: Annotated[Path, typer.Argument(help="新版分层文章评测 JSON")],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """校验实验条件并比较所有隔离层通道。"""
    try:
        baseline = load_layered_article_evaluation(baseline_path)
        current = load_layered_article_evaluation(current_path)
        comparison = compare_layered_article_evaluations(baseline, current)
        if as_json:
            typer.echo(comparison.model_dump_json(indent=2))
            return
        typer.echo(
            f"分层文章反查对比：{comparison.baseline_run_id} -> "
            f"{comparison.current_run_id}"
        )
        for channel in comparison.channels:
            label = (
                f"{channel.layer.value}/{channel.mode.value}/{channel.aggregation.value}"
            )
            mrr = channel.metrics["mrr"]
            hit = channel.metrics["hit_at_1"]
            typer.echo(
                f"{label}: Hit@1 Δ={hit.delta if hit.delta is not None else '不可用'}  "
                f"MRR Δ={mrr.delta if mrr.delta is not None else '不可用'}"
            )
            changed = [item for item in channel.questions if item.status != "unchanged"]
            for item in changed:
                typer.echo(
                    f"  {item.question_id}: {item.baseline_rank} -> "
                    f"{item.current_rank} [{item.status}]"
                )
    except Exception as exc:
        _abort(exc)


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
