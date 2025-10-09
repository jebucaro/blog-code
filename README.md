# Blog Code Repository

Welcome! This repository contains all the code examples from my blog at [jonathanbucaro.com](https://jonathanbucaro.com).

## 📁 Repository Structure

Code is organized by programming language:
- **`python/`** - Python examples

Each folder uses the naming convention: `YYYY-MM-descriptive-slug`

## 🚀 Quick Download

### Using Git (Sparse Checkout)
```bash
# Clone repository without downloading all files
git clone --depth 1 --filter=blob:none --sparse git@github.com:jebucaro/blog-code.git

# Navigate to repository
cd blog-code

# Download only the folder you need
git sparse-checkout set python/2025-10-create-a-knowledge-graph-from-text-with-gemini
```

## 🔗 Blog Posts Index

| Date | Language | Title | Folder | Blog Post |
|------|----------|-------|--------|-----------|
| 2025-10 | Python | Create a Knowledge Graph From Text With Gemini | [Link](./python/2025-10-create-a-knowledge-graph-from-text-with-gemini) | [Read](https://jonathanbucaro.com/blog/create-a-knowledge-graph-from-text-with-gemini/) |


## 📝 License

This code is provided as-is for educational purposes. See individual project folders for specific licenses if applicable.

## 💬 Questions?

If you have questions about any code example, feel free to open an issue or visit the corresponding blog post.
