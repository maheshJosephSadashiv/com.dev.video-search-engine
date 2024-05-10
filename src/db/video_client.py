import math
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from collections import defaultdict

conn = psycopg2.connect(database="postgres",
                        user="postgres",
                        password="postgres",
                        host='0.0.0.0',
                        port=5432)


def createTable(len_of_embedding):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    table_create_command = f"""
    CREATE TABLE IF NOT EXISTS embeddings (
                id bigserial primary key, 
                video_name text,
                time_stamp text,
                frame_num int,
                embedding vector({len_of_embedding})
                );
                """
    cur.execute(table_create_command)
    cur.close()
    conn.commit()


def insertEmbedding(data_frame):
    register_vector(conn)
    cur = conn.cursor()
    # Prepare the list of tuples to insert
    data_list = [(row['video_name'], row['time_stamp'], row['frame_num'], np.array(row['embedding'])) for _, row in data_frame.iterrows()]
    # Use execute_values to perform batch insertion
    execute_values(cur, "INSERT INTO embeddings (video_name, time_stamp, frame_num, embedding) VALUES %s", data_list)
    # Commit after we insert all embeddings
    conn.commit()
    cur.close()


def createIndex():
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM embeddings")
    num_records = cur.fetchone()
    num_lists = num_records[0] / 1000
    if num_lists < 10:
        num_lists = 10
    if num_records[0] > 1000000:
        num_lists = math.sqrt(num_records[0])
    # use the cosine distance measure, which is what we'll later use for querying
    cur.execute(f'CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});')
    conn.commit()


def get_top3_similar_docs(query_embedding, video_name):
    register_vector(conn)
    embedding_array = np.array(query_embedding)
    cur = conn.cursor()
    # Get the top 3 most similar documents using the KNN <=> operator
    cur.execute("SELECT video_name, time_stamp FROM embeddings where video_name = '{}' ORDER BY"
                " embedding <=> %s LIMIT 3".format(video_name), (embedding_array,))
    top3_docs = cur.fetchall()
    return top3_docs

def get_video_name(query_embedding_list):
    register_vector(conn)
    cur = conn.cursor()
    # Get the top 3 most similar documents using the KNN <=> operator
    res = []
    for query_embedding in query_embedding_list:
        embedding_array = np.array(query_embedding)
        cur.execute("SELECT video_name, time_stamp FROM embeddings ORDER BY embedding <=> %s LIMIT 1", (embedding_array,))
        res.append(cur.fetchall())
    dic = defaultdict(int)
    for i in res:
        dic[i[0][0]] += 1
    return max(dic, key=dic.get)