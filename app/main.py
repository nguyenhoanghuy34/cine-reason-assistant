from app.agent.graph import invoke_with_memory


def run_chat(user_id: int, thread_id: str = "default") -> None:
    print("Nhập câu hỏi về phim; nhập /exit để kết thúc.")
    print(f"User {user_id} | Phiên: {thread_id}")
    while True:
        query = input("Bạn: ").strip()
        if query.lower() in {"/exit", "/quit"}:
            return
        if not query:
            continue
        try:
            result = invoke_with_memory(user_id, query, thread_id)
        except Exception as exc:
            print(f"Không hoàn tất lượt hỏi: {exc}")
            continue
        print(f"Assistant: {result.get('response', '')}")


if __name__ == "__main__":
    try:
        while True:
            try:
                user_id = int(input("User ID: ").strip())
                break
            except ValueError:
                print("User ID phải là số nguyên.")
        thread_id = input("Tên phiên (Enter = default, dùng lại tên để tiếp tục): ").strip() or "default"
        run_chat(user_id, thread_id)
    except (KeyboardInterrupt, EOFError):
        print("\nĐã kết thúc hội thoại.")
