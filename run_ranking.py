"""
Runner script for Redrob Intelligence Engine
Processes candidates from JSONL and generates ranked submission
"""

import json
import sys
import os
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from engine import RedrobIntelligenceEngine, run_ranking


def load_job_description():
    """Load the job description from the challenge data"""
    return """Senior AI Engineer — Founding Team

Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

What you'd actually be doing:
The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.

In practical terms, your first 90 days will probably look like:
Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.
Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.
Weeks 9-12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind.

Skills - Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.
- Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.
- Strong Python. Yes really, we care about code quality.
- Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

Skills - Things we'd like you to have but won't reject you for:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space

Disqualifiers:
- Pure research environments (academic labs, research-only roles) without production deployment
- Recent (under 12 months) projects using LangChain to call OpenAI without substantial pre-LLM-era ML production experience
- Senior engineers who haven't written production code in the last 18 months
- Title-chasers (optimizing for Senior → Staff → Principal by switching every 1.5 years)
- Framework enthusiasts with GitHub full of LangChain tutorials
- People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) in entire career
- People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure
- People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation

Location: Pune/Noida-preferred but flexible. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome.
Notice period: We'd love sub-30-day notice. 30+ day notice candidates bar gets higher.
"""


def load_candidates_from_jsonl(filepath):
    """Load candidates from JSONL file"""
    candidates = []
    total_lines = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if line.strip():
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {total_lines}")
                    continue
    print(f"Processed {total_lines} lines, loaded {len(candidates)} valid candidates")
    return candidates


def main():
    print("=" * 60)
    print("REDCROB TALENT INTELLIGENCE ENGINE")
    print("AI-Powered Candidate Ranking System")
    print("=" * 60)
    
    # File paths - point to the actual data location
    candidates_file = Path("C:/Users/govin/.minimax-agent/projects/3/data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    output_file = Path(__file__).parent / "submission.csv"
    
    # Check if candidates file exists
    if not candidates_file.exists():
        print(f"Error: Candidates file not found at {candidates_file}")
        print("Please ensure the candidates.jsonl file is in the data directory")
        return
    
    print(f"\nLoading candidates from {candidates_file}...")
    candidates = load_candidates_from_jsonl(candidates_file)
    print(f"Loaded {len(candidates)} candidates")
    
    # Load job description
    jd_text = load_job_description()
    print("\nJob Description Loaded:")
    print("  Role: Senior AI Engineer — Founding Team")
    print("  Company: Redrob AI")
    print("  Experience: 5-9 years")
    print("  Location: Pune/Noida, India (Hybrid)")
    
    # Initialize engine
    print("\n" + "-" * 60)
    print("Initializing AI Intelligence Engine...")
    engine = RedrobIntelligenceEngine()
    
    # Parse job requirements
    print("Parsing job requirements...")
    engine.job_requirements = engine.parse_job_description(jd_text)
    print(f"  Must-have skills identified: {len(engine.job_requirements.must_have_skills)}")
    print(f"  Disqualifiers configured: Pure research={engine.job_requirements.disqualify_pure_research}, "
          f"Consulting-only={engine.job_requirements.disqualify_consulting_only}")
    
    # Load candidates
    print("\nLoading candidate data...")
    engine.load_candidates(candidates)
    print(f"  {len(candidates)} candidates loaded")
    
    # Run ranking
    print("\n" + "-" * 60)
    print("Running AI-powered ranking analysis...")
    print("  Analyzing: Skills match, Experience, Career progression,")
    print("             Behavioral signals, Availability, Company quality")
    
    rankings = engine.rank_candidates(top_n=1000)
    
    print(f"\n  Ranking complete! Found {len(rankings)} qualified candidates")
    
    # Show top 10
    print("\n" + "=" * 60)
    print("TOP 10 CANDIDATES")
    print("=" * 60)
    
    for rank, (cand_id, score, reasoning) in enumerate(rankings[:20], 1):
        # Get candidate name
        cand_data = next((c for c in candidates if c['candidate_id'] == cand_id), None)
        name = cand_data['profile']['anonymized_name'] if cand_data else "Unknown"
        title = cand_data['profile']['current_title'] if cand_data else "Unknown"
        exp = cand_data['profile']['years_of_experience'] if cand_data else 0
        country = cand_data['profile']['country'] if cand_data else "Unknown"
        
        print(f"\n#{rank}: {name} ({cand_id})")
        print(f"    Title: {title} | Experience: {exp:.1f} years | Location: {country}")
        print(f"    Score: {score:.4f}")
        print(f"    Reasoning: {reasoning}")
    # Save submission (limited to top 100 candidates for challenge submission requirements)
    print("\n" + "-" * 60)
    print(f"Saving submission to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for rank, (cand_id, score, reasoning) in enumerate(rankings[:100], 1):
            reasoning_escaped = reasoning.replace(',', ';').replace('\n', ' ').replace('"', "'")
            f.write(f"{cand_id},{rank},{score:.4f},{reasoning_escaped}\n")
    
    print(f"Successfully saved top 100 ranked candidates!")
    
    # Save results.js
    def generate_outreach_email(candidate_name, title, skills, experience, location, commute_rec):
        subject = f"Founding AI Engineering Role at Redrob - Interview Invitation for {candidate_name}"
        body = f"Hi {candidate_name},\n\n" \
               f"I hope you're doing well.\n\n" \
               f"I'm the lead recruiter for the founding AI team at Redrob. We are building our core intelligence layer from scratch and came across your profile. Your background as a {title} with {experience:.1f} years of experience and your work with {', '.join(skills[:3])} is an exceptional fit for what we need.\n\n" \
               f"Since you are located in {location}, we recommend a hybrid arrangement ({commute_rec}).\n\n" \
               f"Would you be open to a quick 10-15 minute sync this week to discuss our roadmap?\n\n" \
               f"Best regards,\nRedrob Founding AI Recruiter"
        return {"subject": subject, "body": body}

    print("\n" + "-" * 60)
    print("Generating results.js for the UI dashboard...")
    js_candidates = []
    for rank, (cand_id, score, reasoning) in enumerate(rankings, 1):
        analysis = engine.get_detailed_analysis(cand_id)
        if analysis:
            cand_data = analysis.raw_data
            profile = cand_data.get('profile', {})
            skills_list = [s['name'] for s in cand_data.get('skills', [])]
            
            outreach = generate_outreach_email(
                profile.get('anonymized_name', 'Unknown'),
                profile.get('current_title', 'AI Engineer'),
                skills_list,
                profile.get('years_of_experience', 0),
                profile.get('location', 'India'),
                analysis.commute_recommendation
            )
            
            js_cand = {
                "rank": rank,
                "candidate_id": cand_id,
                "score": float(score),
                "reasoning": reasoning,
                "name": profile.get('anonymized_name', 'Unknown'),
                "title": profile.get('current_title', 'AI Engineer'),
                "experience": float(profile.get('years_of_experience', 0)),
                "location": f"{profile.get('location', '')}, {profile.get('country', '')}",
                "skills": skills_list[:8],
                "is_hidden_gem": analysis.is_hidden_gem,
                "commute_recommendation": analysis.commute_recommendation,
                "outreach": outreach,
                "scores_breakdown": {
                    "skills": float(analysis.skill_match_score),
                    "behavior": float(analysis.behavioral_signal_score),
                    "availability": float(analysis.availability_score),
                    "experience": float(analysis.experience_match_score),
                    "career": float(analysis.career_progression_score),
                    "company": float(analysis.company_quality_score),
                    "education": float(analysis.education_score),
                    "location": float(analysis.location_score),
                    "commute": float(analysis.commute_score)
                }
            }
            js_candidates.append(js_cand)
            
    js_file = Path(__file__).parent / "results.js"
    with open(js_file, 'w', encoding='utf-8') as f:
        f.write("const candidatesData = ")
        f.write(json.dumps(js_candidates, indent=2))
        f.write(";\n")
    print(f"Successfully saved results.js to {js_file}!")
    
    # Show score distribution
    print("\n" + "-" * 60)
    print("SCORE DISTRIBUTION")
    print("-" * 60)
    
    scores = [r[1] for r in rankings]
    if scores:
        print(f"  Highest: {max(scores):.4f}")
        print(f"  Lowest:  {min(scores):.4f}")
        print(f"  Average: {sum(scores)/len(scores):.4f}")
        print(f"  Median:  {sorted(scores)[len(scores)//2]:.4f}")
    
    print("\n" + "=" * 60)
    print("RANKING COMPLETE!")
    print("=" * 60)
    
    return rankings


if __name__ == "__main__":
    main()