import jieba
import logging
import cProfile
import sys
import re
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 关闭jieba日志，避免多余提示
jieba.setLogLevel(logging.INFO)

# -------------------------- 文本预处理相关 --------------------------
# 预编译标点正则（只编译1次，避免每次清洗文本重复编译）
PUNCTUATION_REGEX = re.compile(r'[^\w\s]', re.UNICODE)  # 匹配所有非文字/空格字符


@lru_cache(maxsize=512)
def preprocess_text(text):
    """文本清洗→分词→过滤，减少重复计算和中间开销"""
    # 步骤1：先清洗文本（移除标点，减少无效分词）
    cleaned_text = PUNCTUATION_REGEX.sub('', text).strip()
    if not cleaned_text:  # 空文本直接返回空字符串，避免后续无效操作
        return ''

    # 步骤2：结巴分词（用默认精准模式）
    words = jieba.cut(cleaned_text, cut_all=False)

    # 步骤3：过滤空白字符并拼接
    return ' '.join([word for word in words if word.strip()])


# -------------------------- 相似度计算相关 --------------------------
# 全局复用TF-IDF向量器
GLOBAL_VECTORIZER = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",  # 匹配分词后的词语
    max_features=10000  # 限制最大特征数，减少矩阵计算量
)


def calculate_similarity(original_text, plagiarized_text):
    """计算两篇文本的相似度"""
    original_processed = preprocess_text(original_text)
    plagiarized_processed = preprocess_text(plagiarized_text)

    # 转换为TF-IDF矩阵
    tfidf_matrix = GLOBAL_VECTORIZER.fit_transform([original_processed, plagiarized_processed])

    # 计算余弦相似度
    text_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return text_similarity


# -------------------------- 文件操作相关 --------------------------
def read_file(file_path):
    """读取文件内容，包含异常处理"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            return file_handle.read()
    except FileNotFoundError as fnf_error:
        raise Exception(f"文件相关错误: {str(fnf_error)}")


def write_result(file_path, similarity):
    """将查重结果写入文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write(f"{similarity:.2f}")
    except Exception as err:
        raise Exception(f"写入文件出错: {str(err)}")


# -------------------------- 主函数 --------------------------
def main():
    # 检查命令行参数
    if len(sys.argv) != 4:
        print("使用方法: python main.py [原文文件路径] [抄袭版论文文件路径] [结果文件路径]")
        print("示例: python main.py orig.txt orig_0.8_add.txt result.txt")
        sys.exit(1)

    # 从命令行参数获取文件路径
    original_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    result_path = sys.argv[3]
    profile_stats_path = "profile_stats"

    def core_task():
        """核心任务执行函数"""
        original_text = read_file(original_path)
        plagiarized_text = read_file(plagiarized_path)
        text_similarity = calculate_similarity(original_text, plagiarized_text)
        write_result(result_path, text_similarity)
        return text_similarity

    try:
        # 启用性能分析
        with cProfile.Profile() as profile:
            similarity_result = core_task()
            print(f"查重完成，相似度: {similarity_result:.2f}")
            print(f"结果已保存到: {result_path}")
            profile.dump_stats(profile_stats_path)
            print(f"性能分析数据已保存到 {profile_stats_path} 文件")

            # 尝试自动调用snakeviz
            try:
                import subprocess
                subprocess.run(["snakeviz", profile_stats_path])
            except Exception as snakeviz_err:
                print(f"自动调用snakeviz失败，错误信息: {str(snakeviz_err)}")
                print(f"请手动在终端运行: snakeviz {profile_stats_path} 查看可视化报告")
    except Exception as err:
        print(f"程序运行出错: {str(err)}")


if __name__ == "__main__":
    main()
