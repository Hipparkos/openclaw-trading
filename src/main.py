import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv
from data.ibkr_client import IBKRClient


def load_settings(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


async def main() -> None:
    load_dotenv()

    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    settings = load_settings(settings_path)

    client = IBKRClient(settings)

    try:
        await client.stream_market_data()
    
        shutdown_event = asyncio.Event()
        await shutdown_event.wait()

    except asyncio.CancelledError:
        print("\nShutdown signal received.")
    except Exception as e:
        print(f"\nAn error occurred in the execution loop: {e}")
    finally:
        print("Cleaning up connections...")
        client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped manually.")
        sys.exit(0)