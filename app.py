import chainlit as cl
from dotenv import load_dotenv

from job_search import (
    INDIA_SALARY_BRACKETS,
    SUPPORTED_COUNTRIES,
    run_job_search_pipeline,
)
from rag_pipeline import (
    analyze_skill_gap,
    calculate_confidence,
    compare_jds,
    generate_pitch,
    ingest_resume_text,
    parse_jd,
    predict_questions,
)

load_dotenv()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
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
    lines = raw.strip().split("\n")
    output = "### Readiness Score Dashboard\n\n"
    for line in lines:
        if ":" in line and "%" in line:
            key, _, value = line.partition(":")
            value = value.strip()
            try:
                pct = int(value.replace("%", "").strip())
                bars = "█" * (pct // 10) + "░" * (10 - pct // 10)
                output += f"**{key.strip()}**: `{bars}` {pct}%\n\n"
            except Exception:
                output += f"**{key.strip()}**: {value}\n\n"
        elif "SUMMARY:" in line:
            output += f"\n{line.replace('SUMMARY:', '').strip()}"
    return output


def format_comparison_md(jobs: list) -> str:
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


def format_job_search_results(results: dict, job_title: str, city: str, country_name: str) -> str:
    """Format the full job search pipeline output as markdown."""
    jobs = results.get("jobs", [])
    total_found = results.get("total_found", 0)

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
        bars = "█" * (fit // 10) + "░" * (10 - fit // 10)
        salary_line = ""
        if job.get("salary_min") or job.get("salary_max"):
            smin = job.get("salary_min")
            smax = job.get("salary_max")
            if smin and smax:
                salary_line = f"\n**Salary:** {int(smin):,} - {int(smax):,}"
            elif smin:
                salary_line = f"\n**Salary:** {int(smin):,}+"

        output += (
            f"### {i}. {job.get('title', 'Untitled')} - {job.get('company', 'Unknown')}\n"
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
        if job.get("redirect_url"):
            output += f"[**Apply Here**]({job['redirect_url']})\n\n"
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


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "# Welcome to **JobGenie**!\n\n"
            "I'm your RAG-powered job research assistant.\n\n"
            "**Step 1:** Upload your resume PDF to get started."
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

    async with cl.Step(name="Processing your resume") as step:
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

    skill_keywords = [
        "SQL", "Python", "Power BI", "Excel", "LangChain", "Pinecone",
        "Machine Learning", "Tableau", "Pandas", "NumPy", "RAG", "LLM"
    ]
    skills_found = [s for s in skill_keywords if s.lower() in resume_text.lower()]
    skills_display = " - ".join(skills_found[:6]) if skills_found else "Skills detected"
    first_line = resume_text.strip().split("\n")[0].strip()
    name_line = f" - **{first_line}**" if len(first_line) < 40 else ""

    cl.user_session.set("mode", "single")
    cl.user_session.set("compare_jds_list", [])
    cl.user_session.set("compare_mode_active", False)

    await cl.Message(
        content=(
            f"---\n**Resume saved{name_line}**\n"
            f"Detected: `{skills_display}`\n\n---\n\n"
            "**What would you like to do?**\n\n"
            "Click a button below or paste a JD directly:"
        ),
        actions=[
            cl.Action(name="mode_jobsearch", payload={"mode": "jobsearch"}, label="Find Matching Jobs"),
            cl.Action(name="mode_single",  payload={"mode": "single"},  label="Analyze a Single JD"),
            cl.Action(name="mode_compare", payload={"mode": "compare"}, label="Compare Multiple JDs"),
            cl.Action(name="mode_qa",      payload={"mode": "qa"},      label="Ask About My Profile"),
        ],
    ).send()


@cl.action_callback("mode_single")
async def on_single(action: cl.Action):
    cl.user_session.set("mode", "single")
    cl.user_session.set("compare_mode_active", False)
    await cl.Message(content="**Single JD mode active.**\n\nPaste any Job Description below and I'll give you the full analysis.").send()


@cl.action_callback("mode_compare")
async def on_compare(action: cl.Action):
    cl.user_session.set("mode", "compare")
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
    cl.user_session.set("compare_mode_active", False)
    await cl.Message(
        content=(
            "**Q&A mode active.**\n\n"
            "Ask me anything about your profile - salary expectations, "
            "how to explain your career switch, what skills to highlight, etc."
        )
    ).send()


@cl.action_callback("mode_jobsearch")
async def on_jobsearch(action: cl.Action):
    """
    Runs the 5-question job search flow:
    job title -> country -> city -> salary -> search + score + aggregate.
    """
    resume_text = cl.user_session.get("resume_text", "")

    title_res = await cl.AskUserMessage(
        content="What job title or role are you targeting? (e.g. \"Data Analyst\", \"GTM Engineer\")",
        timeout=120,
    ).send()
    if not title_res:
        await cl.Message(content="No response received. Click **Find Matching Jobs** to try again.").send()
        return
    job_title = title_res["output"].strip()

    country_actions = [
        cl.Action(name="country_pick", payload={"code": code}, label=f"{name}")
        for code, name in SUPPORTED_COUNTRIES.items()
    ]
    country_res = await cl.AskActionMessage(
        content="Which country are you targeting?",
        actions=country_actions,
        timeout=120,
    ).send()
    if not country_res:
        await cl.Message(content="No response received. Click **Find Matching Jobs** to try again.").send()
        return
    country_code = country_res.get("payload", {}).get("code", "in")
    country_name = SUPPORTED_COUNTRIES.get(country_code, "India")

    city_res = await cl.AskUserMessage(
        content=f"Which city in {country_name}? (type \"any\" for no preference)",
        timeout=120,
    ).send()
    if not city_res:
        await cl.Message(content="No response received. Click **Find Matching Jobs** to try again.").send()
        return
    city = city_res["output"].strip()

    salary_min = None
    if country_code == "in":
        bracket_actions = [
            cl.Action(name="salary_pick", payload={"value": str(v) if v else ""}, label=k)
            for k, v in INDIA_SALARY_BRACKETS.items()
        ]
        salary_res = await cl.AskActionMessage(
            content="What's your salary expectation?",
            actions=bracket_actions,
            timeout=120,
        ).send()
        if salary_res:
            val = salary_res.get("payload", {}).get("value", "")
            salary_min = int(val) if val else None
    else:
        salary_res = await cl.AskUserMessage(
            content=(
                "Minimum annual salary in local currency? "
                "(type a number, or \"skip\" for no preference)"
            ),
            timeout=120,
        ).send()
        if salary_res:
            raw = salary_res["output"].strip().lower()
            if raw not in ("skip", "", "no preference", "any"):
                try:
                    salary_min = int("".join(c for c in raw if c.isdigit()))
                except Exception:
                    salary_min = None

    async with cl.Step(name="Searching live job listings") as step:
        results = run_job_search_pipeline(
            resume_context=resume_text,
            job_title=job_title,
            country_code=country_code,
            city=city,
            salary_min=salary_min,
        )
        step.output = f"Found {results.get('total_found', 0)} listings, scored top {len(results.get('jobs', []))}."

    output_md = format_job_search_results(results, job_title, city, country_name)
    if results.get("jobs"):
        output_md += "\n\n*Job listings powered by [Adzuna](https://www.adzuna.com/)*"
    await cl.Message(content=output_md).send()

    await cl.Message(
        content="**Search complete!** What's next?",
        actions=[
            cl.Action(name="mode_jobsearch", payload={"mode": "jobsearch"}, label="Search Again"),
            cl.Action(name="mode_qa",        payload={"mode": "qa"},       label="Ask About My Profile"),
        ],
    ).send()


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
    await cl.Message(
        content="Comparison complete! What's next?",
        actions=[
            cl.Action(name="mode_compare", payload={"mode": "compare"}, label="Compare Again"),
            cl.Action(name="mode_qa",      payload={"mode": "qa"},      label="Ask About My Profile"),
        ],
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    jd_text = message.content.strip()
    mode = cl.user_session.get("mode", "single")
    compare_active = cl.user_session.get("compare_mode_active", False)

    if compare_active:
        if len(jd_text.split()) < 30:
            await cl.Message(content="That looks too short for a JD. Please paste the full job description.").send()
            return

        jd_list = cl.user_session.get("compare_jds_list", [])
        jd_list.append(jd_text)
        cl.user_session.set("compare_jds_list", jd_list)
        count = len(jd_list)

        parsed = parse_jd(jd_text)
        role    = parsed.get("JOB_TITLE", f"Role {count}")
        company = parsed.get("COMPANY", "Unknown")

        if count < 3:
            await cl.Message(
                content=f"**JD {count} saved:** {role} at {company}\n\nAdd another or run comparison:",
                actions=[
                    cl.Action(name="add_another_jd", payload={"action": "add"},     label="Add Another JD"),
                    cl.Action(name="run_comparison",  payload={"action": "compare"}, label=f"Run Comparison ({count} JDs)"),
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

    if mode == "qa":
        from langchain_groq import ChatGroq
        resume_text = cl.user_session.get("resume_text", "")
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)
        response = llm.invoke(
            f"You are JobGenie, a career assistant. Answer based on the candidate's resume.\n"
            f"Be specific and practical for the Indian job market (salary in LPA).\n\n"
            f"Resume:\n{resume_text[:3000]}\n\nQuestion: {jd_text}"
        )
        await cl.Message(content=response.content).send()
        return

    if len(jd_text.split()) < 50:
        await cl.Message(
            content=(
                "Please paste a complete Job Description (at least 50 words).\n\n"
                "Or use **Find Matching Jobs** to search live listings, "
                "or switch to **Ask About My Profile** for general questions."
            )
        ).send()
        return

    async with cl.Step(name="Parsing Job Description") as step:
        parsed  = parse_jd(jd_text)
        role    = parsed.get("JOB_TITLE", "N/A")
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
            "**Analysis complete!** What's next?\n\n"
            "- *'Rewrite the pitch more confidently'*\n"
            "- *'Give me 5 more behavioral questions'*\n"
            "- *'What salary should I ask for this role?'*"
        ),
        actions=[
            cl.Action(name="mode_jobsearch", payload={"mode": "jobsearch"}, label="Find Matching Jobs"),
            cl.Action(name="mode_compare", payload={"mode": "compare"}, label="Compare Multiple JDs"),
            cl.Action(name="mode_qa",      payload={"mode": "qa"},      label="Ask About My Profile"),
        ],
    ).send()
