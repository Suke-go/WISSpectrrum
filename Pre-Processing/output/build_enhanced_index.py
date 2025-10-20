#!/usr/bin/env python3
"""
Build enhanced index.json with embeddings and reduced dimensions
"""

import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_paper_with_embedding(paper_path):
    """Load paper file and extract all section embeddings"""
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        embeddings_data = data.get('embeddings', {})

        # Extract embeddings for each section
        sections = ['abstract', 'overview', 'positioning', 'purpose', 'method', 'evaluation']
        result = {}

        for section in sections:
            embedding = embeddings_data.get(section, [])
            if embedding and len(embedding) > 0:
                result[section] = np.array(embedding)

        return result if result else None
    except Exception as e:
        print(f"Error loading {paper_path}: {e}")
        return None

def compute_reduced_dimensions(embeddings_list, n_components=2):
    """Reduce embeddings to 2D using PCA and t-SNE"""
    if not embeddings_list or len(embeddings_list) < 2:
        return None, None

    embeddings = np.array(embeddings_list)

    # First reduce to 50D with PCA (faster for t-SNE)
    print(f"Running PCA: {embeddings.shape} -> 50D")
    pca_50 = PCA(n_components=min(50, embeddings.shape[0], embeddings.shape[1]))
    embeddings_pca50 = pca_50.fit_transform(embeddings)

    # Then reduce to 2D with t-SNE
    print(f"Running t-SNE: 50D -> 2D")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings_pca50)

    # Also compute PCA 2D for comparison
    print(f"Running PCA: {embeddings.shape} -> 2D")
    pca_2 = PCA(n_components=2)
    embeddings_pca2d = pca_2.fit_transform(embeddings)

    return embeddings_2d, embeddings_pca2d

def main():
    base_dir = Path(__file__).parent / "summaries"
    index_path = base_dir / "index.json"

    print(f"Loading index from {index_path}")
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    # Collect all papers and their embeddings by section
    papers_list = []
    embeddings_by_section = {
        'abstract': [],
        'overview': [],
        'positioning': [],
        'purpose': [],
        'method': [],
        'evaluation': []
    }

    print("Loading papers and embeddings...")
    for year_block in index_data['years']:
        year = year_block['year']
        print(f"  Processing year {year}...")

        for paper in year_block['papers']:
            paper_path = base_dir / paper['path']
            embeddings_dict = load_paper_with_embedding(paper_path)

            if embeddings_dict is not None:
                papers_list.append({
                    'year': year,
                    'paper': paper
                })

                # Store embeddings for each section
                for section in embeddings_by_section.keys():
                    if section in embeddings_dict:
                        embeddings_by_section[section].append(embeddings_dict[section])
                    else:
                        # Use None placeholder for missing sections
                        embeddings_by_section[section].append(None)

    print(f"Loaded {len(papers_list)} papers with embeddings")

    # Compute reduced dimensions for each section
    reduced_by_section = {}
    for section, embeddings_list in embeddings_by_section.items():
        # Filter out None values
        valid_embeddings = [e for e in embeddings_list if e is not None]

        if len(valid_embeddings) < 2:
            print(f"  Skipping {section}: not enough papers")
            continue

        print(f"\nReducing dimensions for {section}...")
        embeddings_tsne, embeddings_pca = compute_reduced_dimensions(valid_embeddings)

        if embeddings_tsne is not None:
            # Map back to original indices
            valid_idx = 0
            tsne_mapped = []
            pca_mapped = []

            for emb in embeddings_list:
                if emb is not None:
                    tsne_mapped.append(embeddings_tsne[valid_idx])
                    pca_mapped.append(embeddings_pca[valid_idx])
                    valid_idx += 1
                else:
                    tsne_mapped.append(None)
                    pca_mapped.append(None)

            reduced_by_section[section] = {
                'tsne': tsne_mapped,
                'pca': pca_mapped
            }

    # Add reduced dimensions to papers
    print("\nAdding coordinates to papers...")
    for year_block in index_data['years']:
        for paper in year_block['papers']:
            # Find matching paper in our list
            matching = [i for i, p in enumerate(papers_list)
                       if p['paper']['slug'] == paper['slug'] and p['year'] == year_block['year']]

            if matching:
                idx = matching[0]
                paper['embedding_2d'] = {}

                # Add coordinates for each available section
                for section, reduced_data in reduced_by_section.items():
                    if reduced_data['tsne'][idx] is not None:
                        paper['embedding_2d'][section] = {
                            'tsne': reduced_data['tsne'][idx].tolist(),
                            'pca': reduced_data['pca'][idx].tolist()
                        }

    # Save enhanced index
    output_path = base_dir / "index_enhanced.json"
    print(f"\nSaving enhanced index to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    print("✓ Done!")
    print(f"  Total papers: {len(papers_list)}")
    print(f"  With embeddings: {len(embeddings_list)}")
    print(f"  Output: {output_path}")

if __name__ == '__main__':
    main()
