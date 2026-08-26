"""
AI Cloud Operations Agent
Step 19 - Streamlit UI
"""

import streamlit as st

from src.agent.langgraph_workflow import CloudOperationsLangGraph


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Cloud Operations Agent",
    page_icon="☁️",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("☁️ AI Cloud Operations Agent")

st.markdown(
    "Investigate cloud infrastructure issues using "
    "LangGraph, cloud tools, RAG, historical incident "
    "memory, and evidence-driven analysis."
)

st.divider()


# --------------------------------------------------
# Initialize Graph
# --------------------------------------------------

@st.cache_resource
def get_graph():
    return CloudOperationsLangGraph()


graph = get_graph()


# --------------------------------------------------
# User Input
# --------------------------------------------------

question = st.text_area(
    "Enter your cloud operations question",
    placeholder="Example: Why is instance i-demo-001 unhealthy?",
    height=100,
)


# --------------------------------------------------
# Investigation
# --------------------------------------------------

if st.button(
    "🔍 Investigate",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning(
            "Please enter a cloud operations question."
        )

    else:

        with st.spinner(
            "Investigating the issue..."
        ):

            try:

                result = graph.run(
                    question.strip()
                )

                # ------------------------------------------
                # Status
                # ------------------------------------------

                if result.get("status") == "completed":

                    st.success(
                        "Investigation completed successfully."
                    )

                else:

                    st.error(
                        result.get(
                            "error",
                            "The investigation could not be completed.",
                        )
                    )

                # ------------------------------------------
                # Agent Answer
                # ------------------------------------------

                st.subheader("🤖 Agent Answer")

                final_answer = result.get(
                    "final_answer",
                    "No final answer generated.",
                )

                st.markdown(final_answer)

                # ------------------------------------------
                # Investigation Summary
                # ------------------------------------------

                st.subheader(
                    "📊 Investigation Summary"
                )

                col1, col2, col3, col4 = st.columns(4)

                trace_summary = result.get(
                    "trace_summary",
                    {},
                )

                col1.metric(
                    "Iterations",
                    result.get(
                        "iteration",
                        0,
                    ),
                )

                col2.metric(
                    "Tool Calls",
                    len(
                        result.get(
                            "tool_calls",
                            [],
                        )
                    ),
                )

                col3.metric(
                    "Evidence Items",
                    len(
                        result.get(
                            "evidence",
                            [],
                        )
                    ),
                )

                col4.metric(
                    "Confidence",
                    result.get(
                        "confidence_level",
                        "N/A",
                    ),
                )

                # ------------------------------------------
                # Root Cause
                # ------------------------------------------

                root_cause = result.get(
                    "root_cause",
                    "",
                )

                if root_cause:

                    st.subheader(
                        "🎯 Root Cause"
                    )

                    st.info(root_cause)

                # ------------------------------------------
                # Findings
                # ------------------------------------------

                findings = result.get(
                    "findings",
                    [],
                )

                if findings:

                    st.subheader(
                        "🔎 Findings"
                    )

                    for finding in findings:

                        st.write(
                            f"• {finding}"
                        )

                # ------------------------------------------
                # Evidence
                # ------------------------------------------

                evidence = result.get(
                    "evidence",
                    [],
                )

                if evidence:

                    st.subheader(
                        "📋 Evidence"
                    )

                    for index, item in enumerate(
                        evidence,
                        start=1,
                    ):

                        tool_name = item.get(
                            "tool",
                            "Unknown tool",
                        )

                        with st.expander(
                            f"Evidence {index} — {tool_name}"
                        ):

                            st.json(
                                item.get(
                                    "result",
                                    {},
                                )
                            )

                # ------------------------------------------
                # Confidence
                # ------------------------------------------

                confidence = result.get(
                    "confidence_assessment",
                    {},
                )

                if confidence:

                    st.subheader(
                        "📈 Confidence Assessment"
                    )

                    confidence_col1, confidence_col2 = (
                        st.columns(2)
                    )

                    confidence_col1.metric(
                        "Confidence Level",
                        confidence.get(
                            "confidence_level",
                            "N/A",
                        ),
                    )

                    confidence_col2.metric(
                        "Confidence Score",
                        confidence.get(
                            "confidence_score",
                            "N/A",
                        ),
                    )

                    uncertainty = confidence.get(
                        "uncertainty",
                        "",
                    )

                    if uncertainty:

                        st.caption(
                            f"Uncertainty: {uncertainty}"
                        )

                # ------------------------------------------
                # Operational Knowledge
                # ------------------------------------------

                knowledge = result.get(
                    "knowledge",
                    [],
                )

                if knowledge:

                    st.subheader(
                        "📚 Operational Knowledge"
                    )

                    for item in knowledge:

                        with st.expander(
                            item.get(
                                "source",
                                "Knowledge document",
                            )
                        ):

                            st.write(
                                item.get(
                                    "content",
                                    "",
                                )
                            )

                # ------------------------------------------
                # Historical Incidents
                # ------------------------------------------

                historical = result.get(
                    "historical_incidents",
                    [],
                )

                if historical:

                    st.subheader(
                        "🕘 Historical Incidents"
                    )

                    for incident in historical:

                        incident_id = incident.get(
                            "incident_id",
                            "Unknown",
                        )

                        score = incident.get(
                            "similarity_score",
                            "N/A",
                        )

                        st.write(
                            f"**{incident_id}** "
                            f"(similarity score: {score})"
                        )

                # ------------------------------------------
                # Observability
                # ------------------------------------------

                if trace_summary:

                    st.subheader(
                        "🔭 Observability"
                    )

                    trace_col1, trace_col2, trace_col3, trace_col4 = (
                        st.columns(4)
                    )

                    trace_col1.metric(
                        "Run ID",
                        trace_summary.get(
                            "run_id",
                            "N/A",
                        ),
                    )

                    trace_col2.metric(
                        "Events",
                        trace_summary.get(
                            "event_count",
                            0,
                        ),
                    )

                    trace_col3.metric(
                        "Nodes",
                        trace_summary.get(
                            "node_count",
                            0,
                        ),
                    )

                    trace_col4.metric(
                        "Errors",
                        trace_summary.get(
                            "error_count",
                            0,
                        ),
                    )

            except Exception as exc:

                st.error(
                    f"Investigation failed: {exc}"
                )