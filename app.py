# app.py
import streamlit as st
import pandas as pd
import json
import time

from model_connectors import get_connector
from silver_generator import annotate_dataframe, TASK_PROMPTS
from metrics import compare_teacher_student, run_student_inference

st.set_page_config(page_title="Universal Distillation Pipeline", layout="wide")
st.title("Universal Dataset Distillation")

# STEP 1 - Upload dataset
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Drag and drop a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.header("Dataset Overview")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("First 5 Samples")
    st.dataframe(df.head(5))
with col2:
    st.subheader("Column Stats")
    stats = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "unique": df.nunique()
    })
    st.dataframe(stats)

# STEP 2 - Teacher model configuration
st.header("Teacher Model (Annotation)")
t_col1, t_col2 = st.columns(2)

with t_col1:
    teacher_source = st.selectbox("Teacher backend", ["groq", "ollama", "lmstudio", "huggingface", "custom"], key="teacher_source")
    text_col = st.selectbox("Column containing text to distill (X features)", df.columns.tolist())
    source_col = st.selectbox("Optional group/category column (y target)", ["None"] + df.columns.tolist())

with t_col2:
    task_type = st.selectbox("Task type", list(TASK_PROMPTS.keys()) + ["Custom"])
    if task_type == "Custom":
        system_prompt = st.text_area("Custom system prompt", height=220, key="custom_prompt_box")
    else:
        default_prompt = TASK_PROMPTS[task_type]
        system_prompt = st.text_area(
            "System prompt (editable)",
            value=default_prompt,
            height=220,
            key=f"prompt_preview_{task_type}"
        )

teacher_kwargs = {}
if teacher_source == "groq":
    teacher_kwargs["api_key"] = st.text_input("Groq API Key", type="password")
    teacher_kwargs["model_name"] = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
elif teacher_source in ("ollama", "lmstudio"):
    default_url = "http://localhost:11434/v1" if teacher_source == "ollama" else "http://localhost:1234/v1"
    teacher_kwargs["base_url"] = st.text_input("Server base URL", value=default_url)
    teacher_kwargs["model_name"] = st.text_input("Model name (as loaded on server)")
elif teacher_source == "huggingface":
    teacher_kwargs["model_name"] = st.text_input("Hugging Face model repo", value="google/flan-t5-large")
elif teacher_source == "custom":
    teacher_kwargs["url"] = st.text_input("Custom API URL")
    teacher_kwargs["api_key"] = st.text_input("API Key (optional)", type="password")
    teacher_kwargs["request_key"] = st.text_input("Request payload key", value="prompt")
    teacher_kwargs["response_key"] = st.text_input("Response JSON key", value="text")

sample_n = st.slider("Max rows to annotate (0 = all)", 0, len(df), min(50, len(df)))

# STEP 3 - Run distillation (annotation)
if "silver_results" not in st.session_state:
    st.session_state.silver_results = None

if st.button("Run Distillation (Annotate Dataset)"):
    try:
        connector = get_connector(teacher_source, **teacher_kwargs)
    except Exception as e:
        st.error(f"Could not initialize teacher connector: {e}")
        st.stop()

    work_df = df if sample_n == 0 else df.head(sample_n)
    progress = st.progress(0)
    status = st.empty()

    def on_progress(i, total, src):
        progress.progress(min(i / total, 1.0))
        status.text(f"Annotating {i}/{total} - source: {src}")

    results = annotate_dataframe(
        df=work_df,
        text_col=text_col,
        source_col=None if source_col == "None" else source_col,
        system_prompt=system_prompt,
        connector=connector,
        progress_callback=on_progress
    )
    st.session_state.silver_results = results
    st.success(f"Annotation complete: {len(results)} rows distilled.")

if st.session_state.silver_results:
    out_df = pd.DataFrame(st.session_state.silver_results)
    st.header("Distilled Dataset (After Annotation)")
    st.dataframe(out_df.head(5))

    jsonl_str = "\n".join(json.dumps(r) for r in st.session_state.silver_results)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download JSONL", data=jsonl_str, file_name="silver_dataset.jsonl", mime="application/jsonl")
    with d2:
        st.download_button("Download CSV", data=out_df.to_csv(index=False), file_name="silver_dataset.csv", mime="text/csv")

    # STEP 4 - Optional student training
    st.header("Optional: Train a Student Model")
    train_toggle = st.checkbox("I want to train a student model on this distilled dataset")

    if train_toggle:
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            student_source = st.selectbox("Student backend", ["huggingface", "ollama", "lmstudio", "custom"], key="student_source")
        with s_col2:
            if student_source == "huggingface":
                student_model_name = st.selectbox(
                    "Open-source student model",
                    ["google/flan-t5-large", "google/flan-t5-base", "Qwen/Qwen2.5-1.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
                )
            else:
                student_model_name = st.text_input("Student model name (as loaded on server)")

        student_kwargs = {"model_name": student_model_name}
        student_system_prompt = system_prompt

        if student_source in ("ollama", "lmstudio"):
            default_url = "http://localhost:11434/v1" if student_source == "ollama" else "http://localhost:1234/v1"
            student_kwargs["base_url"] = st.text_input("Student server base URL", value=default_url, key="student_url")

            st.markdown("**Prompting Mode (no weight training — inference-only)**")
            prompt_mode = st.radio(
                "Choose evaluation mode",
                ["Zero-shot (use teacher prompt as-is)", "Few-shot (add examples)"],
                key="prompt_mode_local"
            )

            if prompt_mode.startswith("Few-shot"):
                n_shots = st.number_input("Number of few-shot examples to inject", 1, 10, 3, key="n_shots_local")
                shot_rows = out_df.sample(n=min(n_shots, len(out_df)), random_state=1)
                few_shot_block = "\n\nEXAMPLES:\n"
                for _, r in shot_rows.iterrows():
                    few_shot_block += f"\nINPUT: {r['input'][:400]}\nOUTPUT: {r['output']}\n"
                student_system_prompt = system_prompt + few_shot_block
                st.text_area("Preview: system prompt with injected examples", value=student_system_prompt, height=200, key="fewshot_preview")
            else:
                st.text_area("Preview: zero-shot system prompt (same as teacher)", value=student_system_prompt, height=150, key="zeroshot_preview")

        elif student_source == "custom":
            student_kwargs["url"] = st.text_input("Custom student API URL", key="student_custom_url")
            student_kwargs["api_key"] = st.text_input("Student API Key", type="password", key="student_custom_key")

            st.markdown("**Prompting Mode (no weight training — inference-only)**")
            prompt_mode = st.radio(
                "Choose evaluation mode",
                ["Zero-shot (use teacher prompt as-is)", "Few-shot (add examples)"],
                key="prompt_mode_custom"
            )

            if prompt_mode.startswith("Few-shot"):
                n_shots = st.number_input("Number of few-shot examples to inject", 1, 10, 3, key="n_shots_custom")
                shot_rows = out_df.sample(n=min(n_shots, len(out_df)), random_state=1)
                few_shot_block = "\n\nEXAMPLES:\n"
                for _, r in shot_rows.iterrows():
                    few_shot_block += f"\nINPUT: {r['input'][:400]}\nOUTPUT: {r['output']}\n"
                student_system_prompt = system_prompt + few_shot_block
                st.text_area("Preview: system prompt with injected examples", value=student_system_prompt, height=200, key="fewshot_preview_custom")
            else:
                st.text_area("Preview: zero-shot system prompt (same as teacher)", value=student_system_prompt, height=150, key="zeroshot_preview_custom")
        else:
            st.caption("Hugging Face path uses real fine-tuning below (weights are actually updated), not prompting.")

        max_eval = min(50, len(out_df))
        min_eval = min(2, max_eval)
        if max_eval <= min_eval:
            eval_n = max_eval
            st.info(f"Only {len(out_df)} annotated row(s) available — using all {eval_n} for comparison.")
        else:
            eval_n = st.slider("Held-out samples for teacher vs. student comparison", min_eval, max_eval, min(10, max_eval))

        do_finetune = False
        if student_source == "huggingface":
            do_finetune = st.checkbox("Actually fine-tune this model on the distilled dataset (not just zero-shot eval)", value=True)
            if do_finetune:
                ft_col1, ft_col2, ft_col3 = st.columns(3)
                with ft_col1:
                    epochs = st.number_input("Epochs", 1, 20, 3)
                with ft_col2:
                    batch_size = st.number_input("Batch size", 1, 32, 4)
                with ft_col3:
                    lr = st.number_input("Learning rate", 1e-6, 1e-2, 3e-4, format="%.6f")

        target_field = st.text_input("Target field to compare (e.g. claim, label) - leave blank to skip field-level accuracy", value="claim")

        if st.button("Train / Connect Student & Compare Metrics"):
            trained_model_dir = None

            if student_source == "huggingface" and do_finetune:
                from student_generic import DistillConfig, train_student

                jsonl_path = "silver_dataset_temp.jsonl"
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for r in st.session_state.silver_results:
                        f.write(json.dumps(r) + "\n")

                cfg = DistillConfig(
                    model_name=student_model_name,
                    silver_jsonl=jsonl_path,
                    output_dir="./student_model_trained",
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    lr=float(lr)
                )

                train_progress = st.progress(0)
                train_status = st.empty()

                def on_train_progress(epoch, total_epochs, train_loss, val_loss):
                    train_progress.progress(min(epoch / total_epochs, 1.0))
                    train_status.text(f"Epoch {epoch}/{total_epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

                with st.spinner("Fine-tuning student model... this may take a while"):
                    try:
                        trained_model_dir = train_student(cfg, progress_callback=on_train_progress)
                        st.success(f"Training complete. Model saved to {trained_model_dir}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.stop()

                student_kwargs["model_name"] = trained_model_dir

            with st.spinner("Connecting to student model..."):
                try:
                    student_connector = get_connector(student_source, **student_kwargs)
                except Exception as e:
                    st.error(f"Could not initialize student connector: {e}")
                    st.stop()

            eval_df = out_df.sample(n=eval_n, random_state=42).reset_index(drop=True)
            teacher_outputs = eval_df["output"].tolist()
            raw_texts = eval_df["input"].tolist()

            with st.spinner("Running student model on held-out samples..."):
                student_outputs = run_student_inference(student_connector, student_system_prompt, raw_texts)

            summary, comparison_df = compare_teacher_student(teacher_outputs, student_outputs, target_field=target_field or None)

            st.subheader("Teacher vs. Student Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Similarity (whole JSON)", summary["avg_similarity"])
            m2.metric("Exact Match Rate (whole JSON)", summary["exact_match_rate"])
            m3.metric("Teacher Valid JSON %", summary["teacher_valid_json_rate"])
            m4.metric("Student Valid JSON %", summary["student_valid_json_rate"])

            if target_field and f"{target_field}_accuracy" in summary:
                st.subheader(f"Field-Level Accuracy on '{target_field}'")
                fa1, fa2 = st.columns(2)
                fa1.metric(f"{target_field} Accuracy", summary[f"{target_field}_accuracy"])
                fa2.metric("Comparable Rows", summary[f"{target_field}_comparable_rows"])
                st.caption("This compares only the specific target field (e.g. claim label), ignoring free-text fields like justification, which rarely match word-for-word even when both models agree.")

            st.subheader("Per-Sample Comparison")
            st.dataframe(comparison_df)