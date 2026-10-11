# Blog Code Repository

Welcome! This repository contains all the code examples from my blog at [jonathanbucaro.com](https://jonathanbucaro.com).

## 📁 Repository Structure

Code is organized by programming language:
- **`python/`** - Python examples

## 🚀 Quick Download

### Using Git (Sparse Checkout)
```bash
# Clone repository without downloading all files
git clone --depth 1 --filter=blob:none --sparse git@github.com:jebucaro/blog-code.git

# Navigate to repository
cd blog-code

# Download only the folder you need
git sparse-checkout set nodus
```

## 🔗 Blog Posts Index

Here's a list of all the code examples and their corresponding blog posts.

| Date      | Language | Title                                                                                             | Folder                |
|-----------|----------|---------------------------------------------------------------------------------------------------|-----------------------|
| 2025-10   | Python   | [Create a Knowledge Graph From Text With Gemini](https://jonathanbucaro.com/blog/create-a-knowledge-graph-from-text-with-gemini/) | [Nodus](./nodus) |
| 2026-10   | HTML     | AWS Community Day Guatemala 2026 (post pending) | [AWS Community Day GT 2026](./aws-community-day-gt-2026) |


## 👨‍💻 About the Author

My name is Jonathan Bucaro, and I'm a systems engineer who loves to write about technology. You can find more of my work on my [blog](https://jonathanbucaro.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/jonathanbucaro/).


## 📝 License

This code is provided as-is for educational purposes. See individual project folders for specific licenses if applicable.

## 💬 Questions?

If you have questions about any code example, feel free to open an issue or visit the corresponding blog post.
