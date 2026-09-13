"""Interactive CLI for the grounded movie assistant."""

from backend.agent.graph import MovieAssistantGraph
from backend.agent.tools import MovieToolbox
from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository


def main() -> None:
	agent = MovieAssistantGraph(
		MovieToolbox(MovieRepository(MovieLensDataLoader().load()))
	)

	try:
		user_id = int(input("User ID [1]: ").strip() or "1")
	except ValueError:
		print("User ID must be an integer.")
		return

	print("Movie assistant ready. Type 'exit' or 'quit' to stop.")
	while True:
		try:
			query = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break

		if query.casefold() in {"exit", "quit"}:
			break
		if not query:
			continue

		try:
			state = agent.invoke(user_id, query)
			print(f"Assistant: {state['final_answer']}\n")
		except Exception as exc:
			print(f"Assistant error: {exc}\n")


if __name__ == "__main__":
	main()
