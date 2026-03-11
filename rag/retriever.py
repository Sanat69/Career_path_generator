from rag.embedder import embed_text, get_collection
from models import UserProfile


def build_profile_text(profile: UserProfile) -> str:
    """
    Convert a structured profile into a natural language string
    for embedding. This is what gets compared against career docs.
    """
    parts = [
        f"Current role: {profile.current_role} in {profile.current_industry}.",
        f"Experience: {profile.years_of_experience} years.",
        f"Education: {profile.highest_degree} in {profile.field_of_study} from a {profile.institution_tier} institution.",
        f"Technical skills: {', '.join(profile.technical_skills)}.",
        f"Soft skills: {', '.join(profile.soft_skills)}.",
    ]

    if profile.certifications:
        parts.append(f"Certifications: {', '.join(profile.certifications)}.")

    parts.append(f"Interested in: {', '.join(profile.interest_domains)}.")
    parts.append(f"Career goal: {profile.career_goal}.")
    parts.append(f"Life stage: {profile.life_stage}, age {profile.age}.")

    if profile.burnout_level >= 6:
        parts.append(f"Experiencing burnout (level {profile.burnout_level}/10).")

    if profile.recent_life_event != "None":
        parts.append(f"Recent life event: {profile.recent_life_event}.")

    if profile.has_dependents:
        parts.append("Has dependents to consider.")

    parts.append(f"Priority: {profile.work_life_priority}.")
    parts.append(f"Target timeline: {profile.target_timeline_years} years.")

    return " ".join(parts)


def _build_domain_filter(profile: UserProfile) -> dict | None:
    """
    Build a ChromaDB where filter to prefer documents matching
    the user's interest domains. Returns None if no useful filter.
    """
    domains = profile.interest_domains
    if not domains:
        return None

    # Map common interest domain names to career doc domain names
    domain_map = {
        "AI/ML": "AI & ML",
        "Cybersecurity": "Cybersecurity",
        "Product Management": "Product Management",
        "EdTech": "EdTech & Technical Education",
        "Cloud & DevOps": "Cloud & DevOps",
        "FinTech": "FinTech & Banking Technology",
        "Healthcare IT": "Healthcare IT & Health Tech",
        "Digital Marketing": "Digital Marketing & Growth",
        "UI/UX Design": "UI/UX Design",
        "Data Engineering": "Data Analytics & Business Intelligence",
        "Full Stack Development": "Full Stack Web Development",
        "Entrepreneurship": "Entrepreneurship & Startups",
        "Research & Academia": "Research & Academia",
        "Consulting": "Consulting & Strategy",
    }

    mapped_domains = []
    for d in domains:
        mapped = domain_map.get(d, d)
        mapped_domains.append(mapped)

    if len(mapped_domains) == 1:
        return {"domain": {"$eq": mapped_domains[0]}}
    else:
        return {"domain": {"$in": mapped_domains}}


def retrieve_career_docs(
    profile: UserProfile,
    top_k: int = 5,
) -> list[dict]:
    """
    Given a user profile, retrieve the top-K most relevant career
    documents from ChromaDB using semantic similarity + domain filtering.

    Returns list of dicts: [{doc_id, text, metadata, distance}]
    """
    collection = get_collection()
    profile_text = build_profile_text(profile)
    profile_embedding = embed_text(profile_text)

    # Step 1: Try filtered retrieval (matching interest domains)
    domain_filter = _build_domain_filter(profile)
    results = None

    if domain_filter:
        try:
            results = collection.query(
                query_embeddings=[profile_embedding],
                n_results=top_k,
                where=domain_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"Filtered query failed ({e}), falling back to unfiltered.")
            results = None

    # If filtered returned too few results, do unfiltered as fallback
    if not results or not results["ids"][0] or len(results["ids"][0]) < 3:
        print("Using unfiltered retrieval (not enough domain-specific docs).")
        results = collection.query(
            query_embeddings=[profile_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    # Parse results into clean dicts
    docs = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append({
                "doc_id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

    print(f"Retrieved {len(docs)} documents for profile '{profile.full_name}'")
    for d in docs:
        print(f"  - {d['doc_id']} [{d['metadata'].get('domain')}] "
              f"({d['metadata'].get('doc_type')}) dist={d['distance']:.4f}")

    return docs
