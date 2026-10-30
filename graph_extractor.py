import os
import re
import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set
import chromadb
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOP_WORDS = {
    'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'were', 'which',
    'they', 'what', 'their', 'will', 'would', 'there', 'about', 'when', 'make',
    'like', 'time', 'just', 'know', 'take', 'person', 'into', 'year', 'your',
    'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
    'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after',
    'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new',
    'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are',
    'was', 'were', 'been', 'has', 'had', 'does', 'did', 'page', 'notes', 'table',
    'chapter', 'section', 'figure', 'example', 'author', 'title', 'summary',
    'overview', 'introduction', 'conclusion', 'document', 'pdf', 'text', 'line'
}

COMMON_ACRONYMS = {
    'llm': 'LLM', 'slm': 'SLM', 'rag': 'RAG', 'gpu': 'GPU', 'lpu': 'LPU',
    'cpu': 'CPU', 'api': 'API', 'jvm': 'JVM', 'ram': 'RAM', 'sdk': 'SDK',
    'nlp': 'NLP', 'bert': 'BERT', 'gpt': 'GPT', 'ann': 'ANN', 'knn': 'KNN',
    'sql': 'SQL', 'nosql': 'NoSQL', 'rest': 'REST', 'json': 'JSON', 'http': 'HTTP'
}

# Recognized entity categories
ENTITY_COLORS = {
    "document": "#06b6d4",   # Cyan
    "technology": "#38bdf8", # Sky Blue
    "concept": "#818cf8",    # Indigo
    "process": "#ec4899",    # Pink
    "component": "#a855f7",  # Violet
    "chunk": "#64748b"       # Slate
}

CONNECTORS = {'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'the', 'a', 'an'}
PRONOUNS = {'it', 'them', 'they', 'this', 'that', 'these', 'those', 'we', 'you', 'he', 'she', 'its', 'their', 'our'}

def normalize_entity_name(raw_name: str) -> str:
    """
    Cleans, strips noise, trims dangling prepositions, and normalizes casing for an entity name.
    Preserves established technical acronyms.
    """
    clean = re.sub(r'[\(\)\[\]\{\}\"\'\:\,\;\.\?\!]', ' ', raw_name)
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    if not clean or len(clean) < 2:
        return ""
    
    words = clean.split()
    
    # Trim leading connectors
    while words and words[0].lower() in CONNECTORS:
        words.pop(0)
    # Trim trailing connectors
    while words and words[-1].lower() in CONNECTORS:
        words.pop()
        
    if not words or len(words) > 4:
        return ""
        
    # Check if any word is a pronoun
    if any(w.lower() in PRONOUNS for w in words):
        return ""
        
    full_lower = " ".join(words).lower()
    if full_lower in STOP_WORDS or full_lower in PRONOUNS:
        return ""
        
    if full_lower in COMMON_ACRONYMS:
        return COMMON_ACRONYMS[full_lower]
        
    if len(words) == 1 and re.fullmatch(r'[A-Z0-9]{2,6}', words[0]):
        return words[0]
        
    normalized_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in COMMON_ACRONYMS:
            normalized_words.append(COMMON_ACRONYMS[w_lower])
        elif len(w) >= 2 and w.isupper():
            normalized_words.append(w)
        elif w_lower in ('of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'):
            normalized_words.append(w_lower)
        else:
            normalized_words.append(w.capitalize())
            
    return " ".join(normalized_words)

def is_valid_entity(name: str) -> bool:
    """Validates that a string is a meaningful entity and not garbage."""
    if not name or len(name) < 2 or len(name) > 45:
        return False
    if re.fullmatch(r'[\d\W_]+', name):
        return False
    words = name.lower().split()
    if not words:
        return False
    if all(w in STOP_WORDS or w in PRONOUNS for w in words):
        return False
    if any(w in PRONOUNS for w in words):
        return False
    return True

def is_valid_predicate(predicate: str) -> bool:
    """Validates relationship predicate."""
    if not predicate or len(predicate) < 2 or len(predicate) > 35:
        return False
    if re.fullmatch(r'[\d\W_]+', predicate):
        return False
    return True

def extract_triples_via_groq(
    text_chunks: List[Dict[str, Any]], 
    doc_name: str, 
    api_key: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts structured Subject-Predicate-Object triples with source provenance using Groq LLM.
    Samples representative text chunks to keep execution fast (<3s).
    """
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except Exception as e:
        logger.warning(f"Groq client initialization failed: {e}")
        return [], []

    # Limit to top representative chunks across pages
    sampled = text_chunks[:8] if len(text_chunks) <= 8 else (
        [text_chunks[0]] + 
        text_chunks[1:len(text_chunks)-1:max(1, len(text_chunks)//6)] + 
        [text_chunks[-1]]
    )

    combined_excerpts = []
    for i, c in enumerate(sampled):
        page = c.get("page", 1)
        cid = c.get("id", f"c_{i}")
        snippet = c.get("text", "")[:400].replace("\n", " ").strip()
        combined_excerpts.append(f"[Chunk {cid} | Page {page}]: {snippet}")

    prompt = f"""You are a Knowledge Graph specialist. Analyze the following document excerpts from "{doc_name}" and extract core entities and factual Subject-Predicate-Object relationships.

Excerpts:
{chr(10).join(combined_excerpts)}

CRITICAL REQUIREMENTS:
1. Extract 8 to 15 key entities. For each entity provide:
   - "name": Concise canonical name (e.g. "Java Virtual Machine", "Heap Memory", "Self-Attention")
   - "type": One of ["concept", "technology", "process", "component"]
   - "description": 1-sentence definition based on the text.
2. Extract 8 to 18 verified Subject-Predicate-Object triples.
   - Do NOT invent relationships. The relationship must be explicitly stated in the text.
   - "predicate": Specific semantic action verb (e.g. "uses", "implements", "allocates", "contains", "requires", "produces", "extends", "manages").
   - "evidence": The exact sentence or clause from the excerpts proving this relationship.
   - "chunk_id": The chunk identifier where this was stated.
   - "page": The page number integer where this was stated.

Return strictly valid JSON with this exact schema:
{{
  "entities": [
    {{"name": "...", "type": "concept", "description": "..."}}
  ],
  "triples": [
    {{"subject": "...", "predicate": "...", "object": "...", "evidence": "...", "chunk_id": "...", "page": 1}}
  ]
}}
"""

    # Dynamically select an available chat model
    model_to_use = "openai/gpt-oss-20b"
    try:
        models_data = client.models.list().data
        available = [m.id for m in models_data if not m.id.startswith("whisper") and "guard" not in m.id]
        if "openai/gpt-oss-20b" in available:
            model_to_use = "openai/gpt-oss-20b"
        elif "openai/gpt-oss-120b" in available:
            model_to_use = "openai/gpt-oss-120b"
        elif available:
            model_to_use = available[0]
    except Exception:
        pass

    try:
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a specialized Knowledge Graph information extraction engine. Return only strictly valid JSON. Do not write introductory conversational text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        content = completion.choices[0].message.content or "{}"
        
        # Extract JSON from potential markdown fences or surrounding text
        json_str = content
        fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                json_str = content[start:end+1]

        parsed = json.loads(json_str)
        entities = parsed.get("entities", [])
        triples = parsed.get("triples", [])
        logger.info(f"Groq successfully extracted {len(entities)} entities and {len(triples)} SPO triples using {model_to_use}")
        return entities, triples
    except Exception as e:
        logger.warning(f"Groq LLM extraction failed: {e}")
        return [], []

def extract_triples_rule_based(
    text_chunks: List[Dict[str, Any]], 
    doc_name: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fallback extractor using grammatical verb patterns when no LLM key is present.
    Extracts explicit grammatical patterns (e.g. 'X is a Y', 'X uses Y', 'X contains Y')
    instead of arbitrary co-occurrences.
    """
    entities_map: Dict[str, Dict[str, Any]] = {}
    triples: List[Dict[str, Any]] = []

    # Common grammatical relation patterns: (regex, predicate, type)
    patterns = [
        (re.compile(r'([A-Z][a-zA-Z0-9\s]{2,25})\s+(?:is an?|are an?)\s+([A-Z][a-zA-Z0-9\s]{2,25})', re.IGNORECASE), "is_a"),
        (re.compile(r'([A-Z][a-zA-Z0-9\s]{2,25})\s+(?:uses|utilizes|relies on)\s+([A-Z][a-zA-Z0-9\s]{2,25})', re.IGNORECASE), "uses"),
        (re.compile(r'([A-Z][a-zA-Z0-9\s]{2,25})\s+(?:contains|consists of|includes)\s+([A-Z][a-zA-Z0-9\s]{2,25})', re.IGNORECASE), "contains"),
        (re.compile(r'([A-Z][a-zA-Z0-9\s]{2,25})\s+(?:allocates|manages|handles)\s+([A-Z][a-zA-Z0-9\s]{2,25})', re.IGNORECASE), "manages"),
        (re.compile(r'([A-Z][a-zA-Z0-9\s]{2,25})\s+(?:produces|generates|outputs)\s+([A-Z][a-zA-Z0-9\s]{2,25})', re.IGNORECASE), "produces"),
        (re.compile(r'([A-Z][a-zA-Z0-9\s]{2,25})\s+(?:implements|extends)\s+([A-Z][a-zA-Z0-9\s]{2,25})', re.IGNORECASE), "implements"),
    ]

    for chunk in text_chunks:
        text = chunk.get("text", "")
        page = chunk.get("page", 1)
        cid = chunk.get("id", "")
        sentences = re.split(r'[\.\n]\s+', text)

        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 15:
                continue

            for pattern, pred in patterns:
                match = pattern.search(sent_clean)
                if match:
                    subj = normalize_entity_name(match.group(1).strip())
                    obj = normalize_entity_name(match.group(2).strip())
                    if (
                        is_valid_entity(subj) and 
                        is_valid_entity(obj) and 
                        subj.lower() != obj.lower()
                    ):
                        if subj not in entities_map:
                            entities_map[subj] = {
                                "name": subj,
                                "type": "concept",
                                "description": sent_clean[:180]
                            }
                        if obj not in entities_map:
                            entities_map[obj] = {
                                "name": obj,
                                "type": "component" if pred in ("contains", "manages") else "concept",
                                "description": sent_clean[:180]
                            }
                        triples.append({
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "evidence": sent_clean,
                            "chunk_id": cid,
                            "page": page
                        })

    return list(entities_map.values()), triples

def build_knowledge_graph(
    persist_base_dir: str, 
    target_doc: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Builds or loads an optimized Knowledge Graph.
    Caches results per document in JSON so opening the graph is instantaneous.
    """
    load_dotenv(override=True)
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    cache_dir = os.path.join(persist_base_dir, ".graph_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Determine which documents to load
    doc_dirs = []
    if os.path.exists(persist_base_dir):
        if target_doc and target_doc != "all":
            target_path = os.path.join(persist_base_dir, target_doc)
            if os.path.isdir(target_path):
                doc_dirs = [target_doc]
        else:
            doc_dirs = [
                d for d in os.listdir(persist_base_dir) 
                if os.path.isdir(os.path.join(persist_base_dir, d)) and not d.startswith(".")
            ]

    # Check cache for single document
    cache_key = f"{target_doc or 'all'}.json"
    cache_file = os.path.join(cache_dir, cache_key)
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                logger.info(f"Loaded knowledge graph from cache: {cache_key}")
                return cached_data
        except Exception as e:
            logger.warning(f"Error reading cache {cache_key}: {e}")

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_ids: Set[str] = set()
    edge_ids: Set[str] = set()

    for doc_name in doc_dirs:
        doc_path = os.path.join(persist_base_dir, doc_name)
        try:
            client = chromadb.PersistentClient(path=doc_path)
            cols = client.list_collections()
            if not cols:
                continue
            collection = cols[0]
            data = collection.get()

            chunk_ids = data.get("ids", [])
            chunk_texts = data.get("documents", [])
            metadatas = data.get("metadatas", []) or [{}] * len(chunk_ids)

            if not chunk_ids:
                continue

            clean_doc_label = doc_name.replace("_", " ").replace("-", " ").title()
            doc_node_id = f"doc_{doc_name}"

            # 1. Add Document Node
            if doc_node_id not in node_ids:
                max_page = max([m.get("page", 0) for m in metadatas if isinstance(m, dict)] or [0]) + 1
                nodes.append({
                    "id": doc_node_id,
                    "label": clean_doc_label,
                    "type": "document",
                    "document": doc_name,
                    "page_count": max_page,
                    "chunk_count": len(chunk_ids),
                    "excerpt": f"Document with {len(chunk_ids)} indexed text chunks across {max_page} pages in ChromaDB.",
                    "size": 26,
                    "color": ENTITY_COLORS["document"]
                })
                node_ids.add(doc_node_id)

            # Package chunks for extraction
            text_chunks = []
            for idx in range(len(chunk_ids)):
                cid = chunk_ids[idx]
                text = chunk_texts[idx]
                meta = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
                text_chunks.append({
                    "id": cid,
                    "index": idx,
                    "text": text,
                    "page": meta.get("page", idx + 1)
                })

            # 2. Add sample of Chunk Nodes for spatial grounding (up to 12 per doc)
            sample_step = max(1, len(text_chunks) // 12)
            sampled_chunks = text_chunks[::sample_step][:12]

            for chunk_obj in sampled_chunks:
                idx = chunk_obj["index"]
                cid = chunk_obj["id"]
                page_num = chunk_obj["page"]
                text = chunk_obj["text"]
                chunk_node_id = f"chunk_{doc_name}_{idx}"

                if chunk_node_id not in node_ids:
                    nodes.append({
                        "id": chunk_node_id,
                        "label": f"Chunk #{idx + 1} (p.{page_num})",
                        "type": "chunk",
                        "document": doc_name,
                        "page": page_num,
                        "chunk_id": cid,
                        "excerpt": (text[:220] + "...").replace("\n", " "),
                        "full_text": text,
                        "size": 8,
                        "color": ENTITY_COLORS["chunk"]
                    })
                    node_ids.add(chunk_node_id)

                # Edge: Document -> Chunk
                doc_chunk_edge = f"e_{doc_node_id}_{chunk_node_id}"
                if doc_chunk_edge not in edge_ids:
                    edges.append({
                        "id": doc_chunk_edge,
                        "source": doc_node_id,
                        "target": chunk_node_id,
                        "relationship": "contains",
                        "predicate": "contains",
                        "color": "rgba(6, 182, 212, 0.25)",
                        "weight": 1
                    })
                    edge_ids.add(doc_chunk_edge)

            # 3. Extract Structured Entities & SPO Triples
            entities: List[Dict[str, Any]] = []
            triples: List[Dict[str, Any]] = []

            if groq_api_key:
                logger.info(f"Running structured Groq extraction for document: {doc_name}")
                entities, triples = extract_triples_via_groq(text_chunks, doc_name, groq_api_key)

            # If LLM didn't return triples or no key, use grammatical fallback
            if not triples:
                logger.info(f"Using rule-based extraction fallback for document: {doc_name}")
                entities_fb, triples_fb = extract_triples_rule_based(text_chunks, doc_name)
                if not entities:
                    entities = entities_fb
                triples = triples_fb

            # 4. Normalize & Deduplicate Entities
            entity_node_map: Dict[str, str] = {}
            for ent in entities:
                raw_name = ent.get("name", "")
                norm_name = normalize_entity_name(raw_name)
                if not is_valid_entity(norm_name):
                    continue

                slug = re.sub(r'[^a-zA-Z0-9]', '_', norm_name).lower()
                node_id = f"ent_{slug}"
                entity_node_map[norm_name.lower()] = node_id

                ent_type = ent.get("type", "concept").lower()
                if ent_type not in ENTITY_COLORS:
                    ent_type = "concept"

                if node_id not in node_ids:
                    nodes.append({
                        "id": node_id,
                        "label": norm_name,
                        "type": ent_type,
                        "document": doc_name,
                        "excerpt": ent.get("description", f"Key entity in {clean_doc_label}."),
                        "size": 14,
                        "color": ENTITY_COLORS.get(ent_type, ENTITY_COLORS["concept"]),
                        "frequency": 1
                    })
                    node_ids.add(node_id)
                else:
                    # Increment frequency
                    for n in nodes:
                        if n["id"] == node_id:
                            n["frequency"] = n.get("frequency", 1) + 1
                            break

                # Connect Document -> Entity (mentions)
                doc_ent_edge = f"e_{doc_node_id}_{node_id}"
                if doc_ent_edge not in edge_ids:
                    edges.append({
                        "id": doc_ent_edge,
                        "source": doc_node_id,
                        "target": node_id,
                        "relationship": "mentions",
                        "predicate": "mentions",
                        "color": "rgba(56, 189, 248, 0.25)",
                        "weight": 1
                    })
                    edge_ids.add(doc_ent_edge)

            # 5. Process & Validate Triples into Semantic Edges
            for tr in triples:
                raw_subj = tr.get("subject", "")
                raw_pred = tr.get("predicate", "related_to")
                raw_obj = tr.get("object", "")
                evidence = tr.get("evidence", "")
                page = tr.get("page", 1)
                chunk_id = tr.get("chunk_id", "")

                subj = normalize_entity_name(raw_subj)
                obj = normalize_entity_name(raw_obj)
                pred = raw_pred.lower().strip().replace(" ", "_")

                if (
                    not is_valid_entity(subj) or 
                    not is_valid_entity(obj) or 
                    subj.lower() == obj.lower() or
                    not is_valid_predicate(pred)
                ):
                    continue

                subj_slug = re.sub(r'[^a-zA-Z0-9]', '_', subj).lower()
                obj_slug = re.sub(r'[^a-zA-Z0-9]', '_', obj).lower()
                subj_id = f"ent_{subj_slug}"
                obj_id = f"ent_{obj_slug}"

                # Ensure both entities exist as nodes
                if subj_id not in node_ids:
                    nodes.append({
                        "id": subj_id,
                        "label": subj,
                        "type": "concept",
                        "document": doc_name,
                        "excerpt": evidence or f"Extracted from {clean_doc_label}.",
                        "size": 13,
                        "color": ENTITY_COLORS["concept"],
                        "frequency": 1
                    })
                    node_ids.add(subj_id)

                if obj_id not in node_ids:
                    nodes.append({
                        "id": obj_id,
                        "label": obj,
                        "type": "component",
                        "document": doc_name,
                        "excerpt": evidence or f"Extracted from {clean_doc_label}.",
                        "size": 13,
                        "color": ENTITY_COLORS["component"],
                        "frequency": 1
                    })
                    node_ids.add(obj_id)

                # Add Semantic Edge
                edge_id = f"e_{subj_id}_{pred}_{obj_id}"
                if edge_id not in edge_ids:
                    edges.append({
                        "id": edge_id,
                        "source": subj_id,
                        "target": obj_id,
                        "relationship": pred,
                        "predicate": pred.replace("_", " "),
                        "evidence": evidence,
                        "page": page,
                        "chunk_id": chunk_id,
                        "document": doc_name,
                        "weight": 2,
                        "color": "rgba(236, 72, 153, 0.45)" # Pink/Purple for semantic SPO links
                    })
                    edge_ids.add(edge_id)

        except Exception as e:
            logger.error(f"Error processing Chroma collection for {doc_name}: {e}")
            continue

    # Compile Graph Response
    result = {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "documents_count": len([n for n in nodes if n["type"] == "document"]),
            "concepts_count": len([n for n in nodes if n["type"] in ("concept", "technology", "process", "component")]),
            "chunks_count": len([n for n in nodes if n["type"] == "chunk"]),
            "total_edges": len(edges)
        }
    }

    # Cache graph to disk
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved knowledge graph cache to: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache knowledge graph: {e}")

    return result
