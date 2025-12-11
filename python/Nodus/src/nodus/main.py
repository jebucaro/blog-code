import logging

from app import StreamlitApp


def main():
    """Entry point for the Streamlit application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
