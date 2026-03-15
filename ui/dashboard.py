from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.session_manager import fetch_user_dashboard


def render_user_dashboard(user_id: int) -> None:
    data = fetch_user_dashboard(user_id)

    st.subheader("My Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions", str(data["total_sessions"]))
    c2.metric("Average Technique", f"{float(data['avg_technique']):.1f}%")
    c3.metric("Average Balance", f"{float(data['avg_balance']):.1f}%")
    c4.metric("Average Consistency", f"{float(data['avg_consistency']):.1f}%")

    dist = data.get("shot_distribution", {})
    st.markdown("### Shot Distribution")
    if dist:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(dist.values(), labels=[k.replace("_", " ").title() for k in dist.keys()], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.info("No session data available yet.")

    st.markdown("### Recent Analyses")
    recent = data.get("recent", [])
    if not recent:
        st.info("No recent analyses.")
        return

    table = pd.DataFrame(recent)
    table = table[[
        "timestamp",
        "video_name",
        "shot_type",
        "confidence",
        "technique_score",
        "balance_score",
        "consistency_score",
    ]]
    table = table.rename(
        columns={
            "timestamp": "Timestamp",
            "video_name": "Video",
            "shot_type": "Shot",
            "confidence": "Confidence",
            "technique_score": "Technique",
            "balance_score": "Balance",
            "consistency_score": "Consistency",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
