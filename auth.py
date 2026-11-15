import sqlite3

USERS_DB = "./users.sqlite3"

def authenticate_user(username: str, password: str):
    with sqlite3.connect(USERS_DB) as conn:
        conn.row_factory = sqlite3.Row

        user = conn.execute(
            """
            SELECT username, user_role
            FROM users
            WHERE username = ? AND password = ?
            """,
            (username, password),
        ).fetchone()

    if user is None:
        return None

    return {
        "username": user["username"],
        "user_role": user["user_role"],
    }

# if __name__ == "__main__":
#     with sqlite3.connect(USERS_DB) as conn:
#             conn.row_factory = sqlite3.Row
    
#             cursor = conn.execute("SELECT * FROM users")
            
#             data = cursor.fetchall()
#             columns = [desc[0] for desc in cursor.description]

#     import pandas as pd

#     df = pd.DataFrame(data, columns=columns)
#     print(df.to_string(index=False))