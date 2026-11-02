"""
app.py - JobGenie RAG Assistant (Chainlit UI)

Flow (Nov 2026 redesign, v2):
1. User uploads resume PDF
2. "Processing your resume..." spinner shown during embedding
3. Resume analyzed via Groq -> 3 AI-recommended job titles shown as buttons + "Other"
4. User picks a title (one click) or types custom
5. Country selection (buttons)
6. City input (free text)
7. Job search runs, results shown ranked with fit scores
8. Secondary options shown below results: "Analyze a JD", "Compare JDs", "Ask About Profile"

IMPORTANT ARCHITECTURE NOTE (fixes a real Chainlit bug, GitHub issue #2204):
cl.AskActionMessage and cl.AskUserMessage must NEVER be called from inside a
function invoked by @cl.action_callback. Doing so corrupts the session so
that any action buttons sent afterward in that same chain become permanently
unresponsive, even though the Ask itself works. This bit us once already:
the original version of this file called AskActionMessage (country) and
AskUserMessage (city) from inside the title-pick action callback, which
worked for those two Asks but silently broke every button sent after them
("What's next?" results, etc).

The fix: country and city collection are NOT blocking Asks. They are plain
cl.Message(actions=...) sends, and the user's reply is captured by the
top-level @cl.on_message handler using a "flow_stage" session variable as a
tiny state machine. No Ask* is ever called from within an action_callback
anywhere in this file.
"""

import asyncio

import chainlit as cl
from dotenv import load_dotenv

from config import (
    GROQ_MODEL,
    JOB_STATUS_LABELS,
    PROMPT_QA,
    SKILL_KEYWORDS,
    SUPPORTED_COUNTRIES,
)
from job_search import run_job_search_pipeline
from rag_pipeline import (
    analyze_skill_gap,
    calculate_confidence,
    compare_jds,
    generate_pitch,
    ingest_resume_text,
    parse_jd,
    predict_questions,
    suggest_job_titles,
)
from storage import (
    get_all_seen_job_keys,
    get_or_create_user,
    is_configured as storage_is_configured,
    record_search,
    save_user,
    summarize_tracker,
    update_job_status,
)

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes. Tries PyMuPDF first, falls back to pypdf."""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()


def format_confidence(raw: str) -> str:
    """Format the readiness score into a visual bar chart."""
    lines = raw.strip().split("\n")
    output = "### Readiness Score Dashboard\n\n"
    for line in lines:
        if ":" in line and "%" in line:
            key, _, value = line.partition(":")
            value = value.strip()
            try:
                pct = int(value.replace("%", "").strip())
                bars = "\u2588" * (pct // 10) + "\u2591" * (10 - pct // 10)
                output += f"**{key.strip()}**: `{bars}` {pct}%\n\n"
            except Exception:
                output += f"**{key.strip()}**: {value}\n\n"
        elif "SUMMARY:" in line:
            output += f"\n{line.replace('SUMMARY:', '').strip()}"
    return output


def format_comparison_md(jobs: list) -> str:
    """Format multi-JD comparison as a markdown table."""
    if not jobs:
        return "No results to compare."
    best_idx = max(range(len(jobs)), key=lambda i: jobs[i].get("match_pct", 0))
    header_cols = []
    for i, j in enumerate(jobs):
        title = j.get("JOB_TITLE", f"Role {i+1}")
        company = j.get("COMPANY", "")
        star = "* " if i == best_idx else ""
        header_cols.append(f"{star}**{title}** - {company}")
    header     = "| Metric | "       + " | ".join(header_cols) + " |"
    sep        = "| --- | "         + " | ".join(["---"] * len(jobs)) + " |"
    match_row  = "| Match % | "      + " | ".join([f"**{j.get('match_pct', 0)}%**" for j in jobs]) + " |"
    skills_row = "| Top Skills | "   + " | ".join([" - ".join(j.get("top_skills", [])[:3]) for j in jobs]) + " |"
    gaps_row   = "| Key Gaps | "     + " | ".join([" - ".join(j.get("top_gaps", [])[:2]) for j in jobs]) + " |"
    diff_row   = "| Difficulty | "   + " | ".join([j.get("difficulty", "Medium") for j in jobs]) + " |"
    table = "\n".join([header, sep, match_row, skills_row, gaps_row, diff_row])
    best = jobs[best_idx]
    recommendation = (
        "\n\n---\n### Recommendation\n\n"
        f"**Best fit:** {best.get('JOB_TITLE', '')} at {best.get('COMPANY', '')} "
        f"- **{best.get('match_pct', 0)}% match**\n\n"
        f"{best.get('recommendation', '')}"
    )
    return f"## Side-by-Side Comparison\n\n{table}{recommendation}"


def format_job_search_results(
    results: dict, job_title: str, city: str, country_name: str, new_job_keys: set = None
) -> str:
    """Format the full job search pipeline output as markdown."""
    jobs = results.get("jobs", [])
    total_found = results.get("total_found", 0)
    new_job_keys = new_job_keys or set()

    if not jobs:
        return (
            f"## No Results\n\n"
            f"Couldn't find listings for **{job_title}** in **{city or 'any location'}, {country_name}**. "
            f"Try a broader job title or a different city."
        )

    output = f"## Matching Jobs for \"{job_title}\" in {city or 'Any City'}, {country_name}\n\n"
    output += f"Found {total_found} listings, scored the top {len(jobs)} against your resume:\n\n"

    for i, job in enumerate(jobs, 1):
        fit = job.get("fit_score", 0)
        bars = "\u2588" * (fit // 10) + "\u2591" * (10 - fit // 10)
        salary_line = ""
        if job.get("salary_min") or job.get("salary_max"):
            smin = job.get("salary_min")
            smax = job.get("salary_max")
            if smin and smax:
                salary_line = f"\n**Salary:** {int(smin):,} - {int(smax):,}"
            elif smin:
                salary_line = f"\n**Salary:** {int(smin):,}+"

        key = f"{job.get('company', '')}::{job.get('title', '')}"
        new_tag = " `NEW`" if key in new_job_keys else ""

        output += (
            f"### {i}. {job.get('title', 'Untitled')} - {job.get('company', 'Unknown')}{new_tag}\n"
            f"**Fit Score:** `{bars}` {fit}%  \n"
            f"**Location:** {job.get('location', 'Not specified')}"
            f"{salary_line}\n\n"
            f"{job.get('one_line_reason', '')}\n\n"
        )
        matched = job.get("matched_skills", [])
        missing = job.get("missing_skills", [])
        if matched:
            output += f"**You have:** {', '.join(matched[:5])}\n\n"
        if missing:
            output += f"**Missing:** {', '.join(missing[:5])}\n\n"
        if job.get("apply_link"):
            output += f"[**Apply Here**]({job['apply_link']})\n\n"
        output += "---\n\n"

    missing_summary = results.get("missing_skills_summary", [])
    if missing_summary:
        output += "## What's Holding You Back\n\n"
        output += (
            "Across all the listings above, here's what comes up most often "
            "as a gap, with where to learn it:\n\n"
        )
        for item in missing_summary:
            output += (
                f"**{item['skill']}** - missing in {item['appears_in']} of "
                f"{item['total_jobs']} listings\n"
            )
            output += (
                f"[YouTube]({item['youtube']}) | "
                f"[freeCodeCamp]({item['freecodecamp']}) | "
                f"[Coursera]({item['coursera']})\n\n"
            )

    return output


def build_title_actions(titles: list[str]) -> list[cl.Action]:
    """Build the job-title selection button list, including the 'Other' option."""
    actions = [
        cl.Action(name="title_pick", payload={"title": title}, label=title)
        for title in titles
    ]
    actions.append(
        cl.Action(name="title_other", payload={"title": "other"}, label="Other (type your own)")
    )
    return actions


async def send_country_prompt():
    """Send the country selection buttons. Never called from inside an Ask*."""
    country_actions = [
        cl.Action(name="country_pick", payload={"code": code}, label=name)
        for code, name in SUPPORTED_COUNTRIES.items()
    ]
    await cl.Message(
        content="Which country are you targeting?",
        actions=country_actions,
    ).send()


async def send_next_steps_menu(heading: str = "**What's next?**"):
    """
    Sends the full action menu: Search Again, Analyze a JD, Compare JDs,
    Ask About Profile, View Tracker. Used after every terminal point in the
    conversation (job search results, Q&A answers, JD analysis, JD
    comparisons), not just after a job search, so the user is never stuck
    scrolling up to find these buttons after asking a follow-up question.
    Never called from inside an action_callback chain that has used an
    Ask* method earlier in the same call stack (see the module docstring's
    note on GitHub issue #2204).
    """
    await cl.Message(
        content=heading,
        actions=[
            cl.Action(name="restart_search", payload={"action": "restart"}, label="Search Again"),
            cl.Action(name="mode_single", payload={"mode": "single"}, label="Analyze a Single JD"),
            cl.Action(name="mode_compare", payload={"mode": "compare"}, label="Compare Multiple JDs"),
            cl.Action(name="mode_qa", payload={"mode": "qa"}, label="Ask About My Profile"),
            cl.Action(name="view_tracker", payload={"action": "tracker"}, label="View My Tracker"),
        ],
    ).send()


async def run_search_and_show_results(job_title: str, country_code: str, city: str):
    """
    Runs the search, shows results, then shows the secondary ('What's next?')
    actions. Called ONLY from on_message (never from inside an action_callback),
    so the action buttons it sends remain clickable.
    """
    resume_text = cl.user_session.get("resume_text", "")
    country_name = SUPPORTED_COUNTRIES.get(country_code, "India")

    # The search + scoring pipeline takes 10-15 seconds (one JSearch call,
    # then one Groq call per job). Previously there was no message before
    # this started, so the 10-15 second wait read as the app being frozen.
    # Chainlit only pushes a Step's content to the UI when it exits (or via
    # an explicit await step.update()), not by just setting attributes
    # mid-loop, so a live per-job counter inside the step isn't reliable
    # without further verification. This uses the simpler, already-proven
    # pattern from the resume upload fix: a plain message sent before the
    # slow work starts, so the user sees it immediately regardless of how
    # the step itself renders.
    await cl.Message(
        content=f"Searching live listings for **{job_title}** and scoring each one against your resume, this takes about 10-15 seconds..."
    ).send()

    async with cl.Step(name="Searching live job listings") as step:
        results = run_job_search_pipeline(
            resume_context=resume_text,
            job_title=job_title,
            country_code=country_code,
            city=city,
        )
        step.output = f"Found {results.get('total_found', 0)} listings, scored top {len(results.get('jobs', []))}."

    # ── Job Tracker: figure out which jobs are new since the user's last visit ──
    tracker_record = cl.user_session.get("tracker_record")
    new_job_keys = set()
    if tracker_record is not None:
        previously_seen = get_all_seen_job_keys(tracker_record, exclude_latest=False)
        for job in results.get("jobs", []):
            key = f"{job.get('company', '')}::{job.get('title', '')}"
            if key not in previously_seen:
                new_job_keys.add(key)

    output_md = format_job_search_results(results, job_title, city, country_name, new_job_keys)
    await cl.Message(content=output_md).send()

    # ── Job Tracker: save this search and show status buttons per job ──
    if tracker_record is not None and results.get("jobs"):
        tracker_record = record_search(tracker_record, job_title, country_code, city, results["jobs"])
        cl.user_session.set("tracker_record", tracker_record)
        user_id = cl.user_session.get("user_id")

        # Same fire-and-forget pattern as mark_applied, for the same reason:
        # HuggingFace rate-limits rapid successive commits (confirmed earlier
        # this session), so this save should never block the next message.
        title_part = f"search results for {job_title}"
        asyncio.create_task(_background_save(user_id, tracker_record, title_part))

        # One short message per job, each with its own three status buttons,
        # instead of three long walls of 8 buttons each grouped by status.
        # Previously a user had to cross-reference "Applied: X" against
        # "Interested: X" against "Not interested: X" across three separate
        # messages to find one job's three options; here they sit together,
        # right under that job's own number, matching the numbering already
        # used in the results list above.
        for i, job in enumerate(results["jobs"][:8], 1):
            key = f"{job.get('company', '')}::{job.get('title', '')}"
            short_label = job.get("title", "")[:34]
            await cl.Message(
                content=f"**#{i}** {short_label}:",
                actions=[
                    cl.Action(name="mark_applied", payload={"job_key": key}, label="Applied"),
                    cl.Action(name="mark_interested", payload={"job_key": key}, label="Interested"),
                    cl.Action(name="mark_not_interested", payload={"job_key": key}, label="Not interested"),
                ],
            ).send()

    await send_next_steps_menu()


# ── Chat Start ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    # No separate "Step 1 / Step 2" preview here anymore: it previously
    # restated almost the same sentence as the question that immediately
    # followed it ("Enter your email so I can save your search history"
    # directly above "Enter your email (used only to save...)"), which
    # read as repetitive. The questions below are self-explanatory on
    # their own, so the welcome message just sets context, not a
    # duplicate script of what's about to be asked.
    await cl.Message(
        content=(
            "# Welcome to **JobGenie**!\n\n"
            "I'm your RAG-powered job research assistant. Let's get started."
        )
    ).send()

    # ── Job Tracker: identify the user (manual ID, not a resume hash) ──
    # A resume-content hash was considered but rejected: it silently breaks
    # the moment the user edits even one line of their resume, which makes
    # it unreliable to demo live. A typed ID is predictable and doubles as
    # a believable feature (most real SaaS products ask for an email).
    user_id = None
    returning = False
    tracker_record = None

    if storage_is_configured():
        id_res = await cl.AskUserMessage(
            content="Enter your email (used only to save your job search history on this device):",
            timeout=120,
        ).send()
        if id_res and id_res["output"].strip():
            user_id = id_res["output"].strip()

            # The lookup below is a real network call to HuggingFace (checking
            # whether this email has a saved record) and was previously silent,
            # leaving a multi-second gap with no feedback. cl.Step gives instant
            # visible acknowledgment the moment the email is submitted, before
            # that network call starts.
            async with cl.Step(name="Checking your saved history...") as step:
                tracker_record, returning = get_or_create_user(user_id)
                step.output = "Found a saved record." if returning else "No saved record, starting fresh."

            cl.user_session.set("user_id", user_id)
            cl.user_session.set("tracker_record", tracker_record)

            if returning:
                summary = summarize_tracker(tracker_record)
                counts = summary["counts"]
                await cl.Message(
                    content=(
                        f"**Welcome back!**\n\n"
                        f"You've run {summary['total_searches']} searches and tracked "
                        f"{summary['total_unique_jobs']} jobs so far: "
                        f"{counts['applied']} applied, {counts['interested']} interested, "
                        f"{counts['not_interested']} not interested.\n\n"
                        f"Upload your resume to run a new search, new matches will be flagged."
                    )
                ).send()

    files = await cl.AskFileMessage(
        content="Upload your resume as a PDF:",
        accept=["application/pdf", ".pdf"],
        max_size_mb=5,
        timeout=180,
    ).send()

    if not files:
        await cl.Message(content="No file received. Please refresh and try again.").send()
        return

    resume_file = files[0]

    # Instant acknowledgment the moment the file lands, before the slow
    # embedding model load and Pinecone connection begin. Previously the
    # cl.Step spinner below was the only feedback, and on a cold free-tier
    # container the model load alone can take 10+ seconds with nothing
    # visible changing on screen in that window.
    await cl.Message(content="Got your resume, reading it now...").send()

    async with cl.Step(name="Processing your resume...") as step:
        try:
            with open(resume_file.path, "rb") as f:
                pdf_bytes = f.read()
            resume_text = extract_text_from_pdf(pdf_bytes)

            if not resume_text or len(resume_text.strip()) < 100:
                await cl.Message(
                    content="Could not read your resume. Make sure it's a text-based PDF."
                ).send()
                return

            ingest_resume_text(resume_text)
            cl.user_session.set("resume_text", resume_text)
            step.output = f"{len(resume_text)} characters extracted and embedded."

        except Exception as e:
            await cl.Message(content=f"Error processing resume: {str(e)}").send()
            return

    skills_found = [s for s in SKILL_KEYWORDS if s.lower() in resume_text.lower()]
    skills_display = " - ".join(skills_found[:6]) if skills_found else "Skills detected"
    first_line = resume_text.strip().split("\n")[0].strip()
    name_line = f" - **{first_line}**" if len(first_line) < 40 else ""

    # Initialize session state. flow_stage drives the on_message state machine.
    cl.user_session.set("mode", "jobsearch")
    cl.user_session.set("flow_stage", None)
    cl.user_session.set("pending_job_title", None)
    cl.user_session.set("pending_country_code", None)
    cl.user_session.set("compare_jds_list", [])
    cl.user_session.set("compare_mode_active", False)

    await cl.Message(
        content=(
            f"---\n**Resume saved{name_line}**\n"
            f"Detected: `{skills_display}`\n\n---"
        )
    ).send()

    async with cl.Step(name="Analyzing your resume for job matches...") as step:
        suggested_titles = suggest_job_titles(resume_text)
        step.output = f"Suggested: {', '.join(suggested_titles)}"

    await cl.Message(
        content=(
            "**Based on your resume, here are the best-fit job titles.**\n\n"
            "Pick one to start searching, or type your own:"
        ),
        actions=build_title_actions(suggested_titles),
    ).send()


# ── Title Selection Callbacks (no Ask* calls here) ──────────────────────────

@cl.action_callback("title_pick")
async def on_title_pick(action: cl.Action):
    """User clicked one of the AI-suggested titles. Move to country selection."""
    job_title = action.payload.get("title", "Data Analyst")
    cl.user_session.set("pending_job_title", job_title)
    cl.user_session.set("flow_stage", "awaiting_country_click")
    await cl.Message(content=f"Great, searching for **{job_title}** roles.").send()
    await send_country_prompt()


@cl.action_callback("title_other")
async def on_title_other(action: cl.Action):
    """User wants to type a custom job title. Capture it via on_message, not Ask*."""
    cl.user_session.set("flow_stage", "awaiting_custom_title")
    await cl.Message(
        content="What job title or role are you targeting? (e.g. \"Data Analyst\", \"GTM Engineer\"). Just type it below."
    ).send()


@cl.action_callback("country_pick")
async def on_country_pick(action: cl.Action):
    """User picked a country. Move to city collection."""
    country_code = action.payload.get("code", "in")
    country_name = SUPPORTED_COUNTRIES.get(country_code, "India")
    cl.user_session.set("pending_country_code", country_code)
    cl.user_session.set("flow_stage", "awaiting_city")
    await cl.Message(
        content=f"Selected: **{country_name}**\n\nWhich city? (type \"any\" for no preference)"
    ).send()


# ── Secondary Mode Callbacks ──────────────────────────────────────────────

@cl.action_callback("restart_search")
async def on_restart_search(action: cl.Action):
    """User wants to search again with a different title."""
    resume_text = cl.user_session.get("resume_text", "")
    cl.user_session.set("flow_stage", None)
    async with cl.Step(name="Analyzing resume...") as step:
        suggested_titles = suggest_job_titles(resume_text)
        step.output = f"Suggested: {', '.join(suggested_titles)}"

    await cl.Message(
        content="Pick a job title or type your own:",
        actions=build_title_actions(suggested_titles),
    ).send()


@cl.action_callback("mode_single")
async def on_single(action: cl.Action):
    cl.user_session.set("mode", "single")
    cl.user_session.set("flow_stage", None)
    cl.user_session.set("compare_mode_active", False)
    await cl.Message(
        content="**Single JD mode active.**\n\nPaste any Job Description below and I'll give you the full analysis."
    ).send()


@cl.action_callback("mode_compare")
async def on_compare(action: cl.Action):
    cl.user_session.set("mode", "compare")
    cl.user_session.set("flow_stage", None)
    cl.user_session.set("compare_jds_list", [])
    cl.user_session.set("compare_mode_active", True)
    await cl.Message(
        content=(
            "**Compare Mode activated!**\n\n"
            "Paste your **first JD** below and hit send.\n"
            "After each JD I'll ask if you want to add another or run the comparison."
        )
    ).send()


@cl.action_callback("mode_qa")
async def on_qa(action: cl.Action):
    cl.user_session.set("mode", "qa")
    cl.user_session.set("flow_stage", None)
    cl.user_session.set("compare_mode_active", False)
    await cl.Message(
        content=(
            "**Q&A mode active.**\n\n"
            "Ask me anything about your profile - salary expectations, "
            "how to explain your career switch, what skills to highlight, etc."
        )
    ).send()


# ── Job Tracker Callbacks ───────────────────────────────────────────────────

async def _background_save(user_id: str, tracker_record: dict, title_part: str):
    """
    Runs the actual HuggingFace upload off the main event loop, genuinely,
    not just scheduled-for-later.

    CORRECTED (this was wrong before): wrapping an async function in
    asyncio.create_task does NOT make a synchronous blocking call inside it
    non-blocking. save_user() makes a real blocking HTTP request (via
    huggingface_hub, which uses the synchronous `requests` library). Calling
    it directly here, even from a function scheduled with create_task, still
    freezes the entire single-threaded asyncio event loop for however long
    that request takes, once the task actually runs. Verified directly: a
    simulated 1.5s blocking call froze delivery of three subsequent chat
    messages for the full 1.5s under the old pattern, and took 0.03s under
    this one. This explains the session-wide freeze after a search: once
    HuggingFace's rate limit kicked in (confirmed earlier this session to
    take minutes once triggered), the blocking save froze not just the save
    itself but every message queued after it, including the Applied/
    Interested/Not Interested buttons and the What's Next menu.

    asyncio.to_thread runs the blocking call on a separate OS thread,
    which actually frees the event loop to keep sending messages while the
    slow HuggingFace request runs in the background.
    """
    saved = await asyncio.to_thread(save_user, user_id, tracker_record)
    if not saved:
        await cl.Message(
            content=f"Note: saving **{title_part}** to long-term storage failed in the background. "
                    f"It's marked for this session but may not persist to your next visit."
        ).send()


async def _mark_job_status(job_key: str, status: str, confirm_verb: str):
    """
    Shared logic for mark_applied / mark_interested / mark_not_interested.
    Updates in-memory state and confirms instantly; the slow network save
    to HuggingFace runs as a background task so repeated clicks never block
    the chat (see _background_save for why this was necessary).
    """
    tracker_record = cl.user_session.get("tracker_record")
    user_id = cl.user_session.get("user_id")

    if tracker_record is None or not user_id:
        await cl.Message(content="Tracker isn't set up for this session.").send()
        return

    tracker_record = update_job_status(tracker_record, job_key, status)
    cl.user_session.set("tracker_record", tracker_record)

    title_part = job_key.split("::", 1)[-1] if "::" in job_key else job_key
    await cl.Message(content=f"Marked **{title_part}** as {confirm_verb}.").send()

    # Fire-and-forget: does not block this callback or any future click.
    asyncio.create_task(_background_save(user_id, tracker_record, title_part))


@cl.action_callback("mark_applied")
async def on_mark_applied(action: cl.Action):
    await _mark_job_status(action.payload.get("job_key", ""), "applied", "applied")


@cl.action_callback("mark_interested")
async def on_mark_interested(action: cl.Action):
    await _mark_job_status(action.payload.get("job_key", ""), "interested", "interested")


@cl.action_callback("mark_not_interested")
async def on_mark_not_interested(action: cl.Action):
    await _mark_job_status(action.payload.get("job_key", ""), "not_interested", "not interested")


@cl.action_callback("view_tracker")
async def on_view_tracker(action: cl.Action):
    """Show a summary of everything tracked for this user across all searches."""
    tracker_record = cl.user_session.get("tracker_record")

    if tracker_record is None:
        await cl.Message(
            content="No tracker data for this session. The Job Tracker needs an email entered at the start of the chat."
        ).send()
        return

    summary = summarize_tracker(tracker_record)
    counts = summary["counts"]

    output = (
        f"## Your Job Tracker\n\n"
        f"**{summary['total_unique_jobs']}** unique jobs seen across **{summary['total_searches']}** searches.\n\n"
        f"| Status | Count |\n|---|---|\n"
        f"| {JOB_STATUS_LABELS['applied']} | {counts['applied']} |\n"
        f"| {JOB_STATUS_LABELS['interested']} | {counts['interested']} |\n"
        f"| {JOB_STATUS_LABELS['not_interested']} | {counts['not_interested']} |\n"
        f"| {JOB_STATUS_LABELS['new']} | {counts['new']} |\n\n"
    )

    applied_jobs = []
    for search in tracker_record.get("searches", []):
        for job in search.get("jobs", []):
            if job.get("status") == "applied":
                applied_jobs.append(job)

    if applied_jobs:
        output += "### Jobs You've Applied To\n\n"
        for job in applied_jobs:
            link = f" - [Apply link]({job['apply_link']})" if job.get("apply_link") else ""
            output += f"- **{job.get('title', '')}** at {job.get('company', '')}{link}\n"

    await cl.Message(content=output).send()


# ── Compare Mode Callbacks ─────────────────────────────────────────────────

@cl.action_callback("add_another_jd")
async def on_add_another(action: cl.Action):
    count = len(cl.user_session.get("compare_jds_list", []))
    await cl.Message(content=f"JD {count} saved! Paste your **next JD** below:").send()


@cl.action_callback("run_comparison")
async def on_run_comparison(action: cl.Action):
    jd_list = cl.user_session.get("compare_jds_list", [])
    if len(jd_list) < 2:
        await cl.Message(content="Please add at least 2 JDs before running comparison.").send()
        return

    cl.user_session.set("compare_mode_active", False)
    await cl.Message(content=f"Running comparison across **{len(jd_list)} JDs**... this takes ~30 seconds.").send()

    async with cl.Step(name="Comparing JDs") as step:
        result = compare_jds(jd_list)
        step.output = f"Compared {len(jd_list)} job descriptions."

    if isinstance(result, list) and len(result) > 0:
        md_output = format_comparison_md(result)
        await cl.Message(content=md_output).send()
    else:
        await cl.Message(content="Comparison returned no results. Please try again.").send()

    cl.user_session.set("compare_jds_list", [])
    cl.user_session.set("mode", "single")
    await send_next_steps_menu("**Comparison complete! What's next?**")


# ── Message Handler (state machine for flow_stage, plus everything else) ───

@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()
    flow_stage = cl.user_session.get("flow_stage")

    # ── State machine: custom job title typed in ──
    if flow_stage == "awaiting_custom_title":
        job_title = text
        cl.user_session.set("pending_job_title", job_title)
        cl.user_session.set("flow_stage", "awaiting_country_click")
        await cl.Message(content=f"Great, searching for **{job_title}** roles.").send()
        await send_country_prompt()
        return

    # ── State machine: city typed in, ready to run the search ──
    if flow_stage == "awaiting_city":
        city = text
        job_title = cl.user_session.get("pending_job_title", "")
        country_code = cl.user_session.get("pending_country_code", "in")
        cl.user_session.set("flow_stage", None)
        await run_search_and_show_results(job_title, country_code, city)
        return

    mode = cl.user_session.get("mode", "single")
    compare_active = cl.user_session.get("compare_mode_active", False)
    jd_text = text

    # ── Compare mode: collecting JDs ──
    if compare_active:
        if len(jd_text.split()) < 30:
            await cl.Message(content="That looks too short for a JD. Please paste the full job description.").send()
            return

        jd_list = cl.user_session.get("compare_jds_list", [])
        jd_list.append(jd_text)
        cl.user_session.set("compare_jds_list", jd_list)
        count = len(jd_list)

        parsed = parse_jd(jd_text)
        role = parsed.get("JOB_TITLE", f"Role {count}")
        company = parsed.get("COMPANY", "Unknown")

        if count < 3:
            await cl.Message(
                content=f"**JD {count} saved:** {role} at {company}\n\nAdd another or run comparison:",
                actions=[
                    cl.Action(name="add_another_jd", payload={"action": "add"}, label="Add Another JD"),
                    cl.Action(name="run_comparison", payload={"action": "compare"}, label=f"Run Comparison ({count} JDs)"),
                ],
            ).send()
        else:
            await cl.Message(
                content=f"**JD {count} saved:** {role} at {company}\n\nMaximum 3 JDs reached.",
                actions=[
                    cl.Action(name="run_comparison", payload={"action": "compare"}, label=f"Run Comparison ({count} JDs)"),
                ],
            ).send()
        return

    # ── Q&A mode ──
    if mode == "qa":
        from langchain_groq import ChatGroq
        resume_text = cl.user_session.get("resume_text", "")
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.5)
        prompt = PROMPT_QA.format(resume_text=resume_text[:3000], question=jd_text)
        response = llm.invoke(prompt)
        await cl.Message(content=response.content).send()
        await send_next_steps_menu("**Ask me anything else, or pick an option:**")
        return

    # ── Single JD analysis (default) ──
    if len(jd_text.split()) < 50:
        await cl.Message(
            content=(
                "Please paste a complete Job Description (at least 50 words).\n\n"
                "Or use the buttons above to search for jobs or ask questions about your profile."
            )
        ).send()
        return

    async with cl.Step(name="Parsing Job Description") as step:
        parsed = parse_jd(jd_text)
        role = parsed.get("JOB_TITLE", "N/A")
        company = parsed.get("COMPANY", "N/A")
        step.output = f"Role: {role} at {company}"

    await cl.Message(
        content=(
            f"## Job Summary\n"
            f"**Role:** {role}\n"
            f"**Company:** {company}\n"
            f"**Experience:** {parsed.get('EXPERIENCE', 'N/A')}\n"
            f"**Key Skills:** {parsed.get('TECHNICAL_SKILLS', 'N/A')}"
        )
    ).send()

    async with cl.Step(name="Analyzing skill gap") as step:
        gap = analyze_skill_gap(jd_text)
        step.output = "Gap analysis complete."
    await cl.Message(content=f"## Skill Gap Analysis\n\n{gap}").send()

    async with cl.Step(name="Generating your pitch") as step:
        pitch = generate_pitch(jd_text)
        step.output = "Pitch generated."
    await cl.Message(content=f"## Your 60-Second Pitch\n\n{pitch}").send()

    async with cl.Step(name="Predicting interview questions") as step:
        questions = predict_questions(jd_text)
        step.output = "Questions predicted."
    await cl.Message(content=f"## Predicted Interview Questions\n\n{questions}").send()

    async with cl.Step(name="Calculating readiness score") as step:
        confidence = calculate_confidence(jd_text)
        step.output = "Score calculated."
    await cl.Message(content=format_confidence(confidence)).send()

    await cl.Message(
        content=(
            "**Analysis complete!** You can also just ask a follow-up directly, for example:\n\n"
            "- *'Rewrite the pitch more confidently'*\n"
            "- *'Give me 5 more behavioral questions'*\n"
            "- *'What salary should I ask for this role?'*"
        )
    ).send()
    await send_next_steps_menu()
