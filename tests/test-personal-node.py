import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.nodes.personal import personal_node
from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


def test_personal_node():
    user_id = 1

    query = "What should I watch tonight?"

    try:
        create_user_temp(user_id)

        result = personal_node(
            {
                "user_id": user_id,
                "query": query,
            }
        )

        print("\n" + "=" * 70)
        print("PERSONAL NODE RESULT")
        print("=" * 70)
        print(f"User ID : {user_id}")
        print(f"Query   : {query}")
        print("-" * 70)
        print(result["response"])
        print("=" * 70)

        assert "response" in result
        assert result["response"]

    finally:
        delete_user_temp(user_id)