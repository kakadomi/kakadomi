import jieba
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 关闭jieba的日志输出，避免运行时出现多余提示
jieba.setLogLevel(logging.INFO)


def read_file(file_path):
    """读取文件内容，包含异常处理"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"文件未找到: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件出错: {str(e)}")


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
    # 检查命令行参数是否正确
    if len(sys.argv) != 4:
        print("使用方法: python main.py [原文文件] [抄袭版论文的文件] [答案文件]")
        print("示例: python main.py orig.txt orig_0.8_add.txt result.txt")
        sys.exit(1)

    # 从命令行参数获取文件路径
    original_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    result_path = sys.argv[3]

    try:
        # 执行完整查重流程
        original_text = read_file(original_path)
        plagiarized_text = read_file(plagiarized_path)
        similarity = calculate_similarity(original_text, plagiarized_text)
        write_result(result_path, similarity)
        print(f"查重完成，相似度: {similarity:.2f}，结果已保存到 {result_path}")
    except Exception as e:
        # 捕获并显示所有可能的错误
        print(f"程序运行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
