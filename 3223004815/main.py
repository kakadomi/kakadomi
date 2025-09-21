import jieba
import logging
import cProfile
import pstats
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os  # 用于检查文件是否存在等操作

# 关闭jieba的日志输出，避免运行时出现多余提示
jieba.setLogLevel(logging.INFO)


def read_file(file_path):
    """读取文件内容，包含异常处理"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as fnf_error:
        raise Exception(f"文件相关错误: {str(fnf_error)}")
    except Exception as e:
        raise Exception(f"读取文件时发生未知错误: {str(e)}")


def preprocess_text(text):
    """文本预处理：中文分词并过滤空白字符"""
    words = jieba.cut(text)
    return ' '.join([word for word in words if word.strip()])


def calculate_similarity(original_text, plagiarized_text):
    """计算两篇文本的余弦相似度（核心查重逻辑）"""
    original_processed = preprocess_text(original_text)
    plagiarized_processed = preprocess_text(plagiarized_text)

    # 将分词后的文本转换为TF-IDF向量
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_processed, plagiarized_processed])

    # 计算向量相似度，返回结果
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


def write_result(file_path, similarity):
    """将查重结果写入文件，精确到小数点后两位"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{similarity:.2f}")
    except Exception as e:
        raise Exception(f"写入文件出错: {str(e)}")


def main():

    # 从命令行参数获取文件路径
    original_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    result_path = sys.argv[3]
    # 性能分析数据保存路径
    profile_stats_path = "profile_stats"

    # 定义核心执行函数（用于性能分析）
    def core_task():
        original_text = read_file(original_path)
        plagiarized_text = read_file(plagiarized_path)
        similarity = calculate_similarity(original_text, plagiarized_text)
        write_result(result_path, similarity)
        return similarity

    try:
        # 启用性能分析，结果保存到指定文件
        with cProfile.Profile() as pr:
            similarity = core_task()
            print(f"查重完成，相似度: {similarity:.2f}")
            print(f"结果已保存到: {result_path}")
            # 保存性能数据
            pr.dump_stats(profile_stats_path)
            print(f"性能分析数据已保存到 {profile_stats_path} 文件")
            # 尝试自动调用snakeviz生成可视化报告
            try:
                import subprocess
                subprocess.run(["snakeviz", profile_stats_path])
            except Exception as snakeviz_error:
                print(f"自动调用snakeviz失败，错误信息: {str(snakeviz_error)}")
                print(f"请手动在终端运行: snakeviz {profile_stats_path} 查看可视化报告")
    except Exception as e:
        # 捕获并显示所有可能的错误
        print(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()
