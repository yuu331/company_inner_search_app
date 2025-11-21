"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct
from langchain.schema import Document
import csv


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )

    # チャンク分割を実施（CSVファイルは分割しない）
    splitted_docs = []
    for doc in docs_all:
        # デバッグ：メタデータを確認
        logger.info(f"ドキュメント処理中: file_name={doc.metadata.get('file_name')}, no_split={doc.metadata.get('no_split')}, file_type={doc.metadata.get('file_type')}")
        
        if doc.metadata.get("no_split") == True:
            # CSVファイル（no_split=True）は分割せずそのまま追加
            splitted_docs.append(doc)
            logger.info(f"チャンク分割をスキップ: {doc.metadata.get('file_name')} (文字数: {len(doc.page_content)})")
        else:
            # その他のファイルは通常通り分割
            chunks = text_splitter.split_documents([doc])
            splitted_docs.extend(chunks)
            logger.info(f"チャンク分割実施: {doc.metadata.get('file_name', 'unknown')} → {len(chunks)}チャンク")

    logger.info(f"最終的なドキュメント数: {len(splitted_docs)}")

    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_SEARCH_K})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ロガーを取得
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # csvの場合、検索精度を上げる形式で1つのドキュメントに統合
        if file_extension == ".csv":
            logger.info(f"CSVファイル読み込み開始: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    headers = csv_reader.fieldnames
                    
                    logger.debug(f"CSVヘッダー: {headers}")
                    
                    # 一旦データを読み込んで行数をカウント
                    rows = list(csv_reader)
                    
                    logger.info(f"CSV行数: {len(rows)}行")
                    
                    # 各カラムの全ユニーク値を抽出（検索キーワードとして冒頭に配置）
                    column_values = {}
                    for header in headers:
                        unique_values = set()
                        for row in rows:
                            value = row.get(header, '')
                            if value and value != 'N/A':
                                unique_values.add(value)
                        column_values[header] = sorted(unique_values)
                    
                    # ドキュメント作成（検索精度向上のため、キーワードを冒頭に配置）
                    merged_text = f"===== {file_name} - データ概要 =====\n"
                    merged_text += f"総レコード数: {len(rows)}件\n\n"
                    
                    # 各カラムの値を列挙（検索キーワードとして機能）
                    merged_text += "【データ項目と値の一覧】\n"
                    for header in headers:
                        values = column_values.get(header, [])
                        if len(values) <= 50:  # 値が多すぎる場合は省略
                            merged_text += f"{header}: {', '.join(map(str, values))}\n"
                        else:
                            merged_text += f"{header}: {len(values)}種類の値\n"
                    merged_text += "\n"
                    
                    # 全レコードを詳細に記載（1行形式）
                    merged_text += "===== 全レコードデータ =====\n"
                    for idx, row in enumerate(rows, start=1):
                        # 1行にまとめて記載（検索時に全情報が近くにある方が有利）
                        record_parts = [f"{header}={row.get(header, '')}" for header in headers]
                        merged_text += f"[{idx}] {' | '.join(record_parts)}\n"
                    
                    # メタデータ
                    metadata = {
                        "source": path,
                        "file_name": file_name,
                        "file_type": "csv",
                        "total_rows": len(rows),
                        "columns": ", ".join(headers),
                        "no_split": True  # ←チャンク分割禁止フラグ
                    }
                    
                    merged_doc = Document(page_content=merged_text, metadata=metadata)
                    docs_all.append(merged_doc)
                    
                    # 読み込み成功のログ出力
                    logger.info(f"CSVファイル読み込み成功: {path} (レコード数: {len(rows)})")
                    logger.info(f"作成されたドキュメントの文字数: {len(merged_text)}")
                    logger.info(f"ドキュメント冒頭100文字: {merged_text[:100]}")
                
            except Exception as e:
                logger.error(f"CSVファイルの読み込みエラー: {path}, エラー: {e}", exc_info=True)
                return
        else:
            # csv以外の場合、元のloaderを使用
            logger.info(f"ファイル読み込み開始: {path} (タイプ: {file_extension})")
            try:
                loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
                docs = loader.load()
                docs_all.extend(docs)
                logger.info(f"ファイル読み込み成功: {path} (ドキュメント数: {len(docs)})")
            except Exception as e:
                logger.error(f"ファイル読み込みエラー: {path}, エラー: {e}", exc_info=True)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s