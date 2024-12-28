import pandas as pd
import numpy as np
import gradio as gr


# 加载数据
def load_data():
    df = pd.read_csv("data/songs_features.csv")
    return df


# 计算特征平均值
def calculate_average_features(df, selected_song_names):
    feature_columns = [f"特征{i}" for i in range(1, 65)]

    # 找到选中歌曲的特征
    selected_features_df = df[df["name"].isin(selected_song_names)]

    # 计算平均特征
    average_features = selected_features_df[feature_columns].mean()

    return average_features


# 特征相似度计算函数
def calculate_similarity(average_features, all_songs):
    feature_columns = [f"特征{i}" for i in range(1, 65)]

    # 计算余弦相似度
    similarities = np.dot(
        all_songs[feature_columns].values, average_features.values
    ) / (
        np.linalg.norm(all_songs[feature_columns].values, axis=1)
        * np.linalg.norm(average_features)
    )

    return similarities


def recommend_songs(selected_song_names):
    df = load_data()

    # 计算选中歌曲的平均特征
    average_features = calculate_average_features(df, selected_song_names)
    df_without_selected = df[~df["name"].isin(selected_song_names)]

    similarities = calculate_similarity(average_features, df_without_selected)

    # 获取top5推荐
    similar_indices = np.argsort(similarities)[::-1][:5]
    recommended_songs = df_without_selected.iloc[similar_indices]
    result_df = pd.DataFrame(
        {
            "歌曲名称": recommended_songs["name"],
            "口味相似度": np.round(similarities[similar_indices], 3) * 100,
            "歌手": recommended_songs["artist_name"],
            "作词": recommended_songs["lyricist"],
            "作曲": recommended_songs["composer"],
        }
    )
    return result_df


def get_random_songs():
    df = load_data()
    random_songs = df.sample(n=10)["name"].tolist()
    return random_songs


def create_interface():
    css = """
    .gradio-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
    }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown("# 🎵 音乐推荐系统演示")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 选择你喜欢的歌曲")
                # 复选框组
                song_checkbox_group = gr.CheckboxGroup(
                    choices=get_random_songs(), label="点击选择歌曲"
                )
                recommend_btn = gr.Button("开始推荐", variant="primary")

            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### 推荐结果")
                # 表格展示推荐结果
                result_table = gr.Dataframe(
                    headers=["歌曲名称", "口味相似度", "歌手", "作词", "作曲"],
                    label="为你推荐的歌曲",
                )

        recommend_btn.click(
            fn=recommend_songs, inputs=song_checkbox_group, outputs=result_table
        )
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
