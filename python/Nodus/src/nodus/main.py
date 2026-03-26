import logging

from dotenv import find_dotenv, load_dotenv

from nodus.app import StreamlitApp


def main():
    """Entry point for the Streamlit application."""
    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
