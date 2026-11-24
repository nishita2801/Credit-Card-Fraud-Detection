
import plotly.graph_objects as go
import pandas as pd


def create_confusion_matrix(results, model_name):
    """
    Create an interactive confusion matrix heatmap.
    """

    cm = results[model_name]["Confusion Matrix"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            text=cm,
            texttemplate="%{text:,}",
            colorscale="Blues",
            showscale=True
        )
    )

    fig.update_layout(
        title=f"{model_name} - Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        template="plotly_white",
        height=450
    )

    return fig


def create_roc_curve(results):
    """
    Compare ROC curves of all trained models.
    """

    fig = go.Figure()

    for model_name, metrics in results.items():

        fpr = metrics["FPR"]
        tpr = metrics["TPR"]
        auc = metrics["ROC-AUC"]

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC = {auc:.4f})"
            )
        )

    # Random classifier reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(dash="dash", color="gray")
        )
    )

    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )

    return fig


def create_metrics_table(comparison_df):
    """
    Create a formatted model comparison table.
    """

    display_df = comparison_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ]

    display_df[metric_columns] = (
        display_df[metric_columns].round(4)
    )

    return display_df


def create_class_distribution_chart(class_counts):
    """
    Display legitimate and fraudulent transaction counts.
    """

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Legitimate", "Fraudulent"],
                y=[
                    int(class_counts.get(0, 0)),
                    int(class_counts.get(1, 0))
                ],
                text=[
                    f"{int(class_counts.get(0, 0)):,}",
                    f"{int(class_counts.get(1, 0)):,}"
                ],
                textposition="auto"
            )
        ]
    )

    fig.update_layout(
        title="Transaction Class Distribution",
        xaxis_title="Transaction Type",
        yaxis_title="Number of Transactions",
        template="plotly_white",
        height=400
    )

    return fig
