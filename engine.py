import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# AI/ML core skills that indicate genuine ML engineering experience
AI_CORE_SKILLS = {
    'llm', 'gpt', 'bert', 'transformer', 'rag', 'langchain', 'hugging face',
    'fine-tuning', 'lora', 'qlora', 'peft', 'pytorch', 'tensorflow', 'keras',
    'xgboost', 'lightgbm', 'catboost', 'sklearn', 'scikit',
    'nlp', 'natural language processing', 'text classification', 'ner',
    'named entity recognition', 'sentiment analysis', 'text summarization',
    'machine learning', 'deep learning', 'neural network',
    'embeddings', 'sentence-transformers', 'bge', 'e5', 'openai embeddings',
    'vector database', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'chroma',
    'faiss', 'opensearch', 'elasticsearch', 'hybrid search', 'dense retrieval',
    'bm25', 'hybrid retrieval', 'reranking', 'reranker',
    'recommendation system', 'recommender', 'collaborative filtering',
    'content-based filtering', 'learning to rank', 'learning-to-rank',
    'ctr', 'click-through rate', 'ctr prediction', 'ranking',
    'evaluation', 'ndcg', 'mrr', 'map', 'precision at k', 'recall at k',
    'a/b testing', 'ab testing', 'offline evaluation', 'online evaluation',
    'mlops', 'kubeflow', 'airflow', 'ml pipeline', 'feature store',
    'data pipeline', 'etl', 'spark', 'databricks', 'dbt',
    'prompt engineering', 'prompt tuning', 'in-context learning',
    'llm inference', 'model serving', 'model deployment',
    'computer vision', 'cnn', 'yolo', 'object detection', 'image classification',
    'vision transformer', 'vit', 'gan', 'stable diffusion', 'diffusion model',
    'speech recognition', 'asr', 'tts', 'audio',
    'reinforcement learning', 'rl', 'rlhf', 'ppo', 'reward modeling',
    'agent', 'autonomous agent', 'tool use', 'function calling',
    'knn', 'k-nearest', 'clustering', 'k-means', 'dimensionality reduction',
    'pca', 'umap', 'topic modeling', 'lda', 'lstm', 'rnn', 'gru',
    'attention', 'self-attention', 'multi-head attention',
    'transformer architecture', 'encoder', 'decoder', 'encoder-decoder',
    'tokenization', 'tokenizer', 'bpe', 'wordpiece', 'subword',
    'fastapi', 'gradio', 'streamlit', 'api', 'rest', 'grpc',
    'docker', 'kubernetes', 'k8s', 'deployment', 'serving',
    'github', 'git', 'version control', 'ci/cd',
    'python', 'sql', 'data engineering', 'analytics engineering',
    'gcp', 'aws', 'azure', 'cloud', 'bigquery', 'snowflake', 'databricks',
    'statistics', 'statistical modeling', 'bayesian', 'hypothesis testing',
    'experimental design', 'causal inference', 'ab test',
    'open source', 'publications', 'papers', 'research'
}

# Skills that suggest disqualifying patterns
DISQUALIFY_SKILLS = {
    'marketing', 'seo', 'content writing', 'graphic design', 'photoshop',
    'illustrator', 'adobe', 'figma', 'ux design', 'ui design', 'ux/ui',
    'accounting', 'finance', 'tax', 'audit', 'quickbooks', 'sap',
    'sales', 'customer support', 'operations', 'hr', 'recruitment',
    'project management', 'agile', 'scrum', 'jira', 'confluence',
    'civil engineering', 'mechanical engineering', 'electrical engineering',
    'structural engineering', 'construction', 'architecture',
    'powerpoint', 'excel', 'word', 'microsoft office',
    'sap', 'erp', 'crm', 'salesforce', 'dynamics',
    'six sigma', 'lean', 'process improvement', 'supply chain',
    'legal', 'compliance', 'regulatory', 'audit'
}

# Disqualifying companies (pure services consulting firms)
DISQUALIFY_COMPANIES = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'hcl', 'tech mahindra', 'ltimindtree', 'mphasis', 'cts', 'cts.',
    'genpact', 'igate', 'patni', 'ibps', 'banks', 'insurance'
}

# Preferred companies (product companies)
PREFERRED_COMPANIES = {
    'google', 'meta', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix',
    'uber', 'lyft', 'airbnb', 'stripe', 'shopify', 'salesforce', 'snowflake',
    'databricks', 'openai', 'anthropic', 'cohere', 'hugging face',
    'linkedin', 'twitter', 'x', 'reddit', 'quora', 'pinterest', 'snap',
    'dropbox', 'box', 'slack', 'zoom', 'atlassian', 'jira', 'confluence',
    'twilio', 'sendgrid', 'cloudflare', 'fastly', 'akamai',
    'palantir', 'datadog', 'new relic', 'splunk', 'sumo logic',
    'coinbase', 'binance', 'blockchain', 'crypto',
    'startup', 'early-stage', 'series a', 'series b', 'series c',
    'product company', 'saas', 'fintech', 'edtech', 'healthtech', 'cleantech'
}

# Company size mapping to numerical value
COMPANY_SIZE_MAP = {
    '1-10': 1,
    '11-50': 2,
    '51-200': 3,
    '201-500': 4,
    '501-1000': 5,
    '1001-5000': 6,
    '5001-10000': 7,
    '10001+': 8
}

TIER_SCORES = {
    'tier_1': 4,  # IIT, IIM, top NLU, etc.
    'tier_2': 3,  # NITs, state engineering colleges, etc.
    'tier_3': 2,  # Other engineering colleges
    'tier_4': 1,  # Local colleges
    'unknown': 0
}

PROFICIENCY_SCORES = {
    'beginner': 1,
    'intermediate': 2,
    'advanced': 3,
    'expert': 4
}

WORK_MODE_SCORES = {
    'remote': 1.0,
    'flexible': 1.0,
    'hybrid': 0.8,
    'onsite': 0.6
}


@dataclass
class JobRequirements:
    """Parsed job requirements from JD"""
    experience_range: Tuple[float, float] = (0, 20)  # min, max years
    must_have_skills: Set[str] = field(default_factory=set)
    nice_to_have_skills: Set[str] = field(default_factory=set)
    disqualifiers: Set[str] = field(default_factory=set)
    preferred_location: str = ""
    preferred_work_mode: str = ""
    notice_period_preference: int = 30  # days
    role_level: str = "senior"  # junior, mid, senior, lead
    industry_preference: Set[str] = field(default_factory=set)
    
    # Disqualification criteria
    disqualify_pure_research: bool = True
    disqualify_langchain_only: bool = True
    disqualify_title_chasers: bool = True
    disqualify_consulting_only: bool = True
    disqualify_cv_speech_only: bool = True


@dataclass
class CandidateAnalysis:
    """Analysis results for a candidate"""
    candidate_id: str
    raw_data: Dict
    
    # Dimension scores (0-100)
    skill_match_score: float = 0
    experience_match_score: float = 0
    career_progression_score: float = 0
    behavioral_signal_score: float = 0
    availability_score: float = 0
    company_quality_score: float = 0
    education_score: float = 0
    location_score: float = 0
    commute_score: float = 0
    
    # Final weighted score
    final_score: float = 0
    
    # Detailed breakdown
    matching_skills: List[str] = field(default_factory=list)
    missing_critical_skills: List[str] = field(default_factory=list)
    disqualifying_factors: List[str] = field(default_factory=list)
    positive_signals: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    # Challenge 2, 3, & 4 markers
    is_hidden_gem: bool = False
    commute_recommendation: str = ""
    is_honeypot: bool = False


class RedrobIntelligenceEngine:
    """
    AI-powered candidate ranking engine that goes beyond keyword matching.
    Uses semantic understanding, behavioral signals, and contextual fit analysis.
    """
    
    def __init__(self):
        self.job_requirements = None
        self.candidates = []
        
    def parse_job_description(self, jd_text: str) -> JobRequirements:
        """
        Parse the job description to extract requirements, disqualifiers, and context.
        This uses rule-based extraction with semantic understanding.
        """
        self.jd_text = jd_text
        req = JobRequirements()
        
        # Extract experience range (5-9 years)
        exp_match = re.search(r'(\d+)[–-](\d+)\s*years?', jd_text, re.IGNORECASE)
        if exp_match:
            req.experience_range = (float(exp_match.group(1)), float(exp_match.group(2)))
        
        # Extract must-have skills
        must_have_patterns = [
            r'absolutely need.*?:(.*?)(?:things we\'d like|$)',
            r'required.*?:(.*?)(?:things we\'d like|preferred|$)',
            r'must have.*?:(.*?)(?:nice to have|preferred|$)',
        ]
        
        for pattern in must_have_patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE | re.DOTALL)
            if match:
                skills_text = match.group(1)
                req.must_have_skills.update(self._extract_skills_from_text(skills_text))
        
        # Key must-haves based on JD analysis
        key_skills = [
            'embeddings', 'vector database', 'retrieval', 'ranking', 'llm', 'lora',
            'python', 'pytorch', 'tensorflow', 'nlp', 'evaluation', 'ndcg', 'mrr',
            'search', 'recommendation', 'rag', 'pinecone', 'milvus', 'weaviate'
        ]
        for skill in key_skills:
            if skill.lower() in jd_text.lower():
                req.must_have_skills.add(skill)
        
        # Extract disqualifiers from JD
        disqualifier_patterns = [
            r'pure research environments?',
            r'academic labs?',
            r'research-only roles?',
            r'langchain.*?only',
            r'senior engineer.*?hasn\'t written production code',
            r'title-chasers',
            r'framework enthusiasts',
            r'consulting firms?',
            r'computer vision.*?without.*?nlp',
            r'speech.*?without.*?nlp',
            r'robotics.*?without.*?nlp',
            r'closed-source.*?without.*?validation',
        ]
        
        for pattern in disqualifier_patterns:
            if re.search(pattern, jd_text, re.IGNORECASE):
                if 'research' in pattern:
                    req.disqualify_pure_research = True
                if 'langchain' in pattern:
                    req.disqualify_langchain_only = True
                if 'consulting' in pattern:
                    req.disqualify_consulting_only = True
                if 'computer vision' in pattern or 'speech' in pattern or 'robotics' in pattern:
                    req.disqualify_cv_speech_only = True
        
        # Extract preferred work mode
        work_mode_map = {
            'hybrid': 'hybrid',
            'remote': 'remote',
            'onsite': 'onsite',
            'in-office': 'onsite',
            'flexible': 'flexible'
        }
        for mode, mapped in work_mode_map.items():
            if mode in jd_text.lower():
                req.preferred_work_mode = mapped
                break
        
        # Extract preferred location
        locations = ['pune', 'noida', 'delhi', 'mumbai', 'hyderabad', 'bangalore', 'bengaluru', 'chennai']
        for loc in locations:
            if loc in jd_text.lower():
                req.preferred_location = loc
        
        # Extract notice period preference
        notice_match = re.search(r'(\d+)\s*day.*?notice|sub-?(\d+)\s*day', jd_text, re.IGNORECASE)
        if notice_match:
            if notice_match.group(1):
                req.notice_period_preference = int(notice_match.group(1))
            if notice_match.group(2):
                req.notice_period_preference = int(notice_match.group(2))
        
        return req
    
    def _extract_skills_from_text(self, text: str) -> Set[str]:
        """Extract skills from text using pattern matching"""
        skills = set()
        text_lower = text.lower()
        
        for skill in AI_CORE_SKILLS:
            if skill in text_lower:
                skills.add(skill)
        
        # Extract quoted skills
        quoted = re.findall(r'"([^"]+)"', text)
        skills.update([s.lower() for s in quoted])
        
        return skills
    
    def load_candidates(self, candidates: List[Dict]):
        """Load candidate data"""
        self.candidates = candidates
    
    def analyze_candidate(self, candidate: Dict) -> CandidateAnalysis:
        """
        Comprehensive analysis of a single candidate.
        This is the core intelligence layer.
        """
        analysis = CandidateAnalysis(
            candidate_id=candidate['candidate_id'],
            raw_data=candidate
        )
        
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        education = candidate.get('education', [])
        signals = candidate.get('redrob_signals', {})
        
        # 1. SKILL MATCH ANALYSIS (0-100)
        analysis.skill_match_score = self._analyze_skills(
            candidate['candidate_id'], skills, career, profile.get('summary', '')
        )
        
        # 2. EXPERIENCE MATCH (0-100)
        analysis.experience_match_score = self._analyze_experience(
            profile, self.job_requirements.experience_range
        )
        
        # 3. CAREER PROGRESSION (0-100)
        analysis.career_progression_score = self._analyze_career_progression(career)
        
        # 4. BEHAVIORAL SIGNALS (0-100)
        analysis.behavioral_signal_score = self._analyze_behavioral_signals(signals)
        
        # 5. AVAILABILITY (0-100)
        analysis.availability_score = self._analyze_availability(signals, career)
        
        # 6. COMPANY QUALITY (0-100)
        analysis.company_quality_score = self._analyze_company_quality(career, signals)
        
        # 7. EDUCATION (0-100)
        analysis.education_score = self._analyze_education(education)
        
        # 8. LOCATION (0-100)
        analysis.location_score = self._analyze_location(profile, self.job_requirements.preferred_location)
        
        # 9. METRO COMMUTE & HYBRID SCORE (Challenge 4)
        analysis.commute_score, analysis.commute_recommendation = self._analyze_commute(profile, signals)
        
        # Blend Location and Commute Scores (40% location match, 60% commute optimization)
        analysis.location_score = min(100, 0.4 * analysis.location_score + 0.6 * analysis.commute_score)
        
        # 10. TIER 2/3 HIDDEN GEMS CHECK (Challenge 4)
        analysis.is_hidden_gem = self._check_hidden_gem(education, skills, signals)
        if analysis.is_hidden_gem:
            # Boost education score for Hidden Gems
            analysis.education_score = min(100, analysis.education_score + 20)
        
        # 11. DISQUALIFICATION CHECKS
        self._check_disqualifications(
            candidate, analysis, career, skills, profile, signals
        )
        
        # Calculate final weighted score
        analysis.final_score = self._calculate_final_score(analysis)
        
        # Generate reasoning
        analysis.reasoning = self._generate_reasoning(analysis)
        
        return analysis
    
    def _analyze_skills(self, candidate_id: str, skills: List[Dict], career: List[Dict], 
                        summary: str) -> float:
        """
        Analyze skills with semantic understanding and TF-IDF similarity.
        """
        if not self.job_requirements:
            return 50
        
        skill_scores = []
        matching = []
        missing = []
        
        # Build comprehensive skill list (skills + career descriptions + summary)
        all_skill_text = ' '.join([
            s['name'].lower() for s in skills
        ] + [
            j['description'].lower() for j in career
        ] + [summary.lower()])
        
        # Check must-have skills
        for req_skill in self.job_requirements.must_have_skills:
            skill_found = False
            skill_level = 0
            
            # Check direct skill match
            for s in skills:
                s_name = s['name'].lower()
                if req_skill.lower() in s_name or s_name in req_skill.lower():
                    skill_found = True
                    skill_level = max(skill_level, PROFICIENCY_SCORES.get(s['proficiency'], 1))
            
            # Check in career descriptions
            if not skill_found:
                for job in career:
                    if req_skill.lower() in job.get('description', '').lower():
                        skill_found = True
                        skill_level = max(skill_level, 3)  # Implicitly advanced if in job desc
                        break
            
            # Check in summary
            if not skill_found and req_skill.lower() in summary.lower():
                skill_found = True
                skill_level = max(skill_level, 2)
            
            if skill_found:
                matching.append(req_skill)
                skill_scores.append(skill_level / 4 * 100)  # Normalize to 0-100
            else:
                missing.append(req_skill)
        
        # Bonus for AI core skills (even if not in must-have list)
        ai_bonus = 0
        disqualify_count = 0
        
        for skill in skills:
            skill_name = skill['name'].lower()
            prof = PROFICIENCY_SCORES.get(skill['proficiency'], 1)
            
            # Check if it's an AI core skill
            is_ai_skill = any(
                ai_term in skill_name 
                for ai_term in AI_CORE_SKILLS
            )
            
            if is_ai_skill:
                ai_bonus += prof * 3  # 3 points per AI skill level
            else:
                # Check if it's a disqualifying skill
                is_disqualifying = any(
                    dq in skill_name 
                    for dq in DISQUALIFY_SKILLS
                )
                if is_disqualifying:
                    disqualify_count += 1
        
        # Additional analysis: Check career descriptions for AI/ML context
        ml_experience_years = 0
        for job in career:
            desc = job.get('description', '').lower()
            if any(term in desc for term in [
                'machine learning', 'ml', 'ai', 'data science', 'nlp',
                'deep learning', 'neural network', 'recommendation',
                'search', 'ranking', 'retrieval', 'embeddings'
            ]):
                ml_experience_years += job.get('duration_months', 0) / 12
        
        # Calculate skill match score
        if self.job_requirements.must_have_skills:
            base_score = sum(skill_scores) / len(skill_scores) if skill_scores else 0
        else:
            base_score = 50  # Neutral if no requirements
        
        # Apply bonuses
        bonus_score = min(ai_bonus / 10, 30)  # Cap bonus at 30 points
        
        # Apply penalties
        penalty_score = disqualify_count * 5  # 5 points per disqualifying skill
        
        # ML experience bonus
        ml_bonus = min(ml_experience_years * 5, 15)  # Up to 15 points for ML experience
        
        heuristic_score = base_score + bonus_score + ml_bonus - penalty_score
        heuristic_score = max(0, min(100, heuristic_score))
        
        # Retrieve semantic TF-IDF score
        semantic_sim = 0.0
        if hasattr(self, 'semantic_scores'):
            semantic_sim = self.semantic_scores.get(candidate_id, 0.0)
            
        max_sem = getattr(self, 'max_semantic_score', 1.0)
        semantic_score = (semantic_sim / max_sem) * 100
        
        # Blend: 40% heuristic, 60% semantic similarity
        final_score = 0.4 * heuristic_score + 0.6 * semantic_score
        
        return max(0, min(100, final_score))
    
    def _analyze_experience(self, profile: Dict, 
                           exp_range: Tuple[float, float]) -> float:
        """Analyze if experience level matches requirements"""
        years_exp = profile.get('years_of_experience', 0)
        
        # Perfect match is in range
        if exp_range[0] <= years_exp <= exp_range[1]:
            return 100
        
        # Slight deviation
        if years_exp < exp_range[0]:
            # Less experience - gradual penalty
            deficit = exp_range[0] - years_exp
            return max(0, 100 - deficit * 15)
        else:
            # More experience - smaller penalty
            excess = years_exp - exp_range[1]
            return max(0, 100 - excess * 8)
    
    def _analyze_career_progression(self, career: List[Dict]) -> float:
        """
        Analyze career progression patterns.
        Looks for: role growth, company quality improvement, relevant industry shifts.
        """
        if not career:
            return 0
        
        score = 50  # Base score
        
        # Check for role growth
        titles = [j.get('title', '').lower() for j in career]
        senior_count = sum(1 for t in titles if any(x in t for x in ['senior', 'lead', 'principal', 'staff']))
        
        if senior_count >= 2:
            score += 15
        elif senior_count == 1:
            score += 5
        
        # Check for tenure patterns (avoid job-hoppers)
        tenures = [j.get('duration_months', 0) for j in career]
        avg_tenure = sum(tenures) / len(tenures) if tenures else 0
        
        if avg_tenure >= 24:  # 2+ years average
            score += 15
        elif avg_tenure >= 18:
            score += 10
        elif avg_tenure >= 12:
            score += 5
        else:
            score -= 15  # Job hopper penalty
        
        # Check for progression (early jobs should be less senior)
        if len(career) > 1:
            # More recent roles should be more senior
            progression_good = tenures[0] <= tenures[-1] if len(tenures) > 1 else True
            if progression_good:
                score += 10
        
        # Industry relevance
        relevant_industries = ['software', 'technology', 'saas', 'ai', 'ml', 'data', 'internet']
        industry_match = sum(
            1 for j in career 
            if any(ind in j.get('industry', '').lower() for ind in relevant_industries)
        )
        score += (industry_match / len(career)) * 10 if career else 0
        
        return max(0, min(100, score))
    
    def _analyze_behavioral_signals(self, signals: Dict) -> float:
        """
        Analyze Redrob behavioral signals.
        These are crucial for predicting actual availability.
        """
        if not signals:
            return 30
        
        score = 0
        
        # Profile completeness (10 points max)
        completeness = signals.get('profile_completeness_score', 0)
        score += completeness / 10
        
        # Recruiter response rate (20 points max)
        response_rate = signals.get('recruiter_response_rate', 0)
        score += response_rate * 20
        
        # Response time (10 points max, faster = better)
        response_time = signals.get('avg_response_time_hours', 200)
        if response_time <= 24:
            score += 10
        elif response_time <= 72:
            score += 7
        elif response_time <= 168:  # 1 week
            score += 4
        else:
            score += 1
        
        # Skill assessment scores (10 points max)
        assessments = signals.get('skill_assessment_scores', {})
        if assessments:
            avg_assessment = sum(assessments.values()) / len(assessments)
            score += avg_assessment / 10
        else:
            score -= 5  # No assessments = slight penalty
        
        # Interview completion rate (10 points max)
        interview_rate = signals.get('interview_completion_rate', 0)
        score += interview_rate * 10
        
        # Offer acceptance rate (5 points max, -1 means no history)
        offer_rate = signals.get('offer_acceptance_rate', -1)
        if offer_rate >= 0:
            score += offer_rate * 5
        
        # Connection count (5 points max)
        connections = signals.get('connection_count', 0)
        if connections >= 500:
            score += 5
        elif connections >= 200:
            score += 3
        elif connections >= 100:
            score += 1
        
        # Endorsements (5 points max)
        endorsements = signals.get('endorsements_received', 0)
        if endorsements >= 50:
            score += 5
        elif endorsements >= 20:
            score += 3
        elif endorsements >= 5:
            score += 1
        
        # Verification badges (5 points max)
        verified_email = signals.get('verified_email', False)
        verified_phone = signals.get('verified_phone', False)
        linkedin = signals.get('linkedin_connected', False)
        score += (verified_email + verified_phone + linkedin) * 1.67
        
        # GitHub activity (5 points max, -1 means not linked)
        github = signals.get('github_activity_score', -1)
        if github >= 0:
            score += min(github / 20, 5)
        
        # Search appearance (5 points max)
        searches = signals.get('search_appearance_30d', 0)
        if searches >= 100:
            score += 5
        elif searches >= 50:
            score += 3
        elif searches >= 20:
            score += 1
        
        return max(0, min(100, score))
    
    def _analyze_availability(self, signals: Dict, career: List[Dict]) -> float:
        """
        Analyze candidate availability for hiring.
        """
        score = 70  # Base availability
        
        # Open to work flag
        if signals.get('open_to_work_flag', False):
            score += 15
        else:
            score -= 10  # Not actively looking
        
        # Last active date
        last_active = signals.get('last_active_date', '')
        if last_active:
            try:
                days_since_active = (datetime.now() - datetime.strptime(
                    last_active, '%Y-%m-%d')).days
                
                if days_since_active <= 7:
                    score += 15
                elif days_since_active <= 30:
                    score += 10
                elif days_since_active <= 90:
                    score += 0
                else:
                    score -= 20  # Inactive
            except:
                pass
        
        # Notice period
        notice = signals.get('notice_period_days', 60)
        if notice <= 30:
            score += 10
        elif notice <= 60:
            score += 0
        else:
            score -= 15  # Long notice period
        
        # Applications submitted (too many = might not be serious)
        apps = signals.get('applications_submitted_30d', 0)
        if apps <= 5:
            score += 5
        elif apps <= 15:
            score += 0
        else:
            score -= 10  # Mass applier
        
        # Salary expectations (check if reasonable)
        salary_range = signals.get('expected_salary_range_inr_lpa', {})
        if salary_range:
            # If max salary is defined and reasonable
            if salary_range.get('max', 0) > 0:
                score += 0  # Neutral
        
        return max(0, min(100, score))
    
    def _analyze_company_quality(self, career: List[Dict], 
                                 signals: Dict) -> float:
        """
        Analyze company quality from career history.
        Prefers product companies over consulting firms.
        """
        if not career:
            return 30
        
        scores = []
        
        for job in career:
            company = job.get('company', '').lower()
            company_size = job.get('company_size', '')
            
            score = 50  # Base score
            
            # Check for disqualifying companies
            is_consulting = any(dq in company for dq in DISQUALIFY_COMPANIES)
            is_preferred = any(p in company for p in PREFERRED_COMPANIES)
            
            if is_consulting:
                score = 20
            elif is_preferred:
                score = 80
            else:
                # Neutral company
                score = 50
                
                # Bonus for larger companies
                size_score = COMPANY_SIZE_MAP.get(company_size, 4) * 5
                score += size_score
            
            scores.append(score)
        
        # Weight recent experience more heavily
        if len(scores) > 1:
            # Current job weighted more
            weighted = scores[-1] * 0.4 + sum(scores[:-1]) / len(scores[:-1]) * 0.6
        else:
            weighted = scores[0] if scores else 50
        
        # Check for variety (alternating between consulting and product = concern)
        all_consulting = all(
            any(dq in j.get('company', '').lower() for dq in DISQUALIFY_COMPANIES)
            for j in career
        )
        if all_consulting:
            return 25  # Pure consulting = low score
        
        return weighted
    
    def _analyze_education(self, education: List[Dict]) -> float:
        """
        Analyze education quality.
        """
        if not education:
            return 40  # No education info = neutral
        
        max_tier = 0
        has_advanced = False
        
        for edu in education:
            tier = edu.get('tier', 'unknown')
            tier_score = TIER_SCORES.get(tier, 0)
            max_tier = max(max_tier, tier_score)
            
            degree = edu.get('degree', '').lower()
            if any(x in degree for x in ['phd', 'ph.d', 'doctorate']):
                has_advanced = True
        
        # Base score from best tier
        score = max_tier * 20
        
        # Bonus for advanced degrees
        if has_advanced:
            score += 10
        
        # Relevant field bonus
        relevant_fields = ['computer science', 'data science', 'artificial intelligence', 
                          'machine learning', 'statistics', 'mathematics', 'information technology']
        
        for edu in education:
            field = edu.get('field_of_study', '').lower()
            if any(rf in field for rf in relevant_fields):
                score += 10
                break
        
        return min(100, score)
    
    def _analyze_location(self, profile: Dict, preferred_loc: str) -> float:
        """
        Analyze location match.
        """
        location = profile.get('location', '').lower()
        country = profile.get('country', '').lower()
        
        if not preferred_loc:
            return 70  # No preference
        
        score = 50  # Base
        
        # India preference
        if 'india' in country:
            score += 15
        
        # Specific city match
        if preferred_loc in location:
            score += 25
        elif preferred_loc in ['pune', 'noida', 'delhi', 'ncr'] and any(
            loc in location for loc in ['pune', 'noida', 'delhi', 'gurgaon', 'gurugram', 'ncr']
        ):
            score += 20
        elif preferred_loc in ['hyderabad', 'bangalore', 'mumbai', 'chennai']:
            if any(loc in location for loc in ['hyderabad', 'bangalore', 'bengaluru', 'mumbai', 'chennai']):
                score += 10
        
        return min(100, score)
    
    def _check_disqualifications(self, candidate: Dict, 
                                  analysis: CandidateAnalysis,
                                  career: List[Dict], skills: List[Dict],
                                  profile: Dict, signals: Dict):
        """
        Check for disqualifying factors based on JD criteria.
        """
        disqualifiers = []
        positive_signals = []
        
        # 1. Check for pure research background
        titles = [j.get('title', '').lower() for j in career]
        descriptions = ' '.join([j.get('description', '') for j in career])
        
        is_research_only = all(
            any(x in t for x in ['research', 'researcher', 'scientist', 'academic'])
            for t in titles
        )
        
        # If only research and no product deployment mentioned
        if is_research_only and 'production' not in descriptions.lower():
            disqualifiers.append("Pure research background without production deployment")
        
        # 2. Check for LangChain-only AI experience
        summary = profile.get('summary', '').lower()
        has_real_ml = any(term in descriptions for term in [
            'machine learning', 'production', 'deployed', 'shipped', 
            'recommendation', 'search', 'ranking', 'nlp'
        ])
        
        langchain_only = 'langchain' in summary or 'langchain' in descriptions
        if langchain_only and not has_real_ml:
            disqualifiers.append("Only LangChain experience without substantial ML production work")
        
        # 3. Check for title-chasing (very short tenures throughout)
        tenures = [j.get('duration_months', 0) for j in career]
        if len(tenures) >= 3:
            short_tenures = sum(1 for t in tenures if t < 18)
            if short_tenures >= len(tenures) - 1:
                disqualifiers.append("Career pattern suggests title-chasing (consistent short tenures)")
        
        # 4. Check for consulting-only career
        all_consulting = all(
            any(dq in j.get('company', '').lower() for dq in DISQUALIFY_COMPANIES)
            for j in career
        )
        if all_consulting and len(career) >= 2:
            disqualifiers.append("Entire career at consulting firms without product company experience")
        
        # 5. Check for CV/Speech without NLP
        cv_skills = sum(1 for s in skills if any(
            x in s['name'].lower() 
            for x in ['computer vision', 'cnn', 'yolo', 'object detection', 
                      'speech', 'asr', 'tts', 'audio', 'nlp', 'nlu']
        ))
        nlp_skills = sum(1 for s in skills if any(
            x in s['name'].lower() 
            for x in ['nlp', 'natural language', 'text', 'language model', 'llm', 'transformer']
        ))
        
        if cv_skills > 0 and nlp_skills == 0 and 'nlp' not in descriptions.lower():
            disqualifiers.append("CV/Speech expertise without NLP/IR exposure")
        
        # 6. Check behavioral red flags
        response_rate = signals.get('recruiter_response_rate', 0)
        if response_rate < 0.1:
            disqualifiers.append("Very low recruiter response rate (<10%)")
        
        last_active = signals.get('last_active_date', '')
        if last_active:
            try:
                days_inactive = (datetime.now() - datetime.strptime(
                    last_active, '%Y-%m-%d')).days
                if days_inactive > 180:
                    disqualifiers.append(f"Inactive for {days_inactive} days")
            except:
                pass
        
        # Now add positive signals
        if has_real_ml:
            positive_signals.append("Has shipped ML systems to production")
        
        if not all_consulting and any(
            any(p in j.get('company', '').lower() for p in PREFERRED_COMPANIES)
            for j in career
        ):
            positive_signals.append("Worked at product companies")
        
        if signals.get('open_to_work_flag', False):
            positive_signals.append("Actively open to work")
        
        if signals.get('verified_email', False) and signals.get('verified_phone', False):
            positive_signals.append("Fully verified profile")
        
        # Store results
        analysis.disqualifying_factors = disqualifiers
        analysis.positive_signals = positive_signals
        
        # Check for honeypot
        is_honeypot, honeypot_reasons = self._detect_honeypot(candidate, career, skills, profile, signals)
        if is_honeypot:
            analysis.is_honeypot = True
            analysis.disqualifying_factors.append(f"Honeypot: {'; '.join(honeypot_reasons)}")
            
        # Update score if disqualified
        if analysis.disqualifying_factors:
            analysis.final_score = max(0, analysis.final_score - 40)
            
    def _build_candidate_text(self, candidate: Dict) -> str:
        profile = candidate.get('profile', {})
        skills = candidate.get('skills', [])
        career = candidate.get('career_history', [])
        
        parts = []
        parts.append(profile.get('headline', ''))
        parts.append(profile.get('summary', ''))
        
        skill_names = [s.get('name', '') for s in skills]
        parts.append(' '.join(skill_names))
        
        for job in career:
            parts.append(job.get('title', ''))
            parts.append(job.get('description', ''))
            
        return ' '.join([p for p in parts if p])

    def _compute_semantic_similarities(self, jd_text: str) -> Dict[str, float]:
        """Compute semantic TF-IDF similarity between JD and all candidates"""
        if not jd_text or not self.candidates:
            return {c['candidate_id']: 0.0 for c in self.candidates}
            
        candidate_texts = []
        candidate_ids = []
        for c in self.candidates:
            candidate_texts.append(self._build_candidate_text(c))
            candidate_ids.append(c['candidate_id'])
            
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        tfidf_matrix = vectorizer.fit_transform(candidate_texts)
        jd_vector = vectorizer.transform([jd_text])
        
        similarities = cosine_similarity(tfidf_matrix, jd_vector).flatten()
        return {cid: float(score) for cid, score in zip(candidate_ids, similarities)}

    def _analyze_commute(self, profile: Dict, signals: Dict) -> Tuple[float, str]:
        location = profile.get('location', '').lower()
        country = profile.get('country', '').lower()
        pref_mode = signals.get('preferred_work_mode', 'hybrid')
        relocate = signals.get('willing_to_relocate', False)
        
        score = 70
        rec = "Remote option suitable"
        
        if 'india' not in country and 'toronto' not in location and 'canada' not in country:
            if not any(city in location for city in ['pune', 'noida', 'delhi', 'bangalore', 'mumbai', 'hyderabad', 'chennai']):
                return 40, "Fully remote cadence required"
                
        if any(city in location for city in ['pune', 'noida', 'delhi', 'ncr', 'gurgaon', 'ghaziabad']):
            if pref_mode in ['hybrid', 'flexible']:
                score = 100
                rec = "Optimal Commute: 2 days hybrid (Pune/Noida Metro corridor)"
            elif pref_mode == 'onsite':
                score = 95
                rec = "Onsite cadence: <45 mins daily transit via local expressways"
            else:
                score = 90
                rec = "Fully Remote cadence preferred by candidate"
        elif any(city in location for city in ['bangalore', 'bengaluru', 'mumbai', 'hyderabad', 'chennai']):
            if relocate:
                score = 85
                rec = "Relocation recommended: 15-day relocation allowance (Noida/Pune)"
            else:
                if pref_mode == 'remote':
                    score = 80
                    rec = "Remote work approved: Regional alignment in India"
                else:
                    score = 50
                    rec = "Commute Alert: Relocation reluctant; High transit friction"
        else:
            if relocate:
                score = 80
                rec = "Relocation required: Tier-2 relocation corridor"
            else:
                score = 40
                rec = "Remote work recommended"
                
        return score, rec

    def _check_hidden_gem(self, education: List[Dict], skills: List[Dict], signals: Dict) -> bool:
        has_tier3_or_4 = False
        for edu in education:
            tier = edu.get('tier', 'unknown')
            if tier in ['tier_3', 'tier_4']:
                has_tier3_or_4 = True
                break
        
        if not has_tier3_or_4:
            return False
            
        github_ok = signals.get('github_activity_score', -1) >= 70
        
        assessments = signals.get('skill_assessment_scores', {})
        assessment_ok = False
        if assessments:
            avg_assess = sum(assessments.values()) / len(assessments)
            if avg_assess >= 75:
                assessment_ok = True
                
        endorsements_ok = signals.get('endorsements_received', 0) >= 25
        
        return github_ok or assessment_ok or endorsements_ok

    def _detect_honeypot(self, candidate: Dict, career: List[Dict], skills: List[Dict], profile: Dict, signals: Dict) -> Tuple[bool, List[str]]:
        reasons = []
        
        expert_zero_duration = 0
        for s in skills:
            if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
                expert_zero_duration += 1
        if expert_zero_duration >= 5:
            reasons.append(f"Expert in {expert_zero_duration} skills with 0 months of experience")
            
        current_year = 2026
        current_month = 6
        for job in career:
            start_date_str = job.get('start_date')
            if start_date_str:
                try:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                    max_months = (current_year - start_date.year) * 12 + (current_month - start_date.month)
                    dur_months = job.get('duration_months', 0)
                    if dur_months > max_months + 3:
                        reasons.append(f"Job at {job.get('company')} lists duration of {dur_months} months, but start date is {start_date_str} (max possible: {max_months} months)")
                except:
                    pass
                    
            end_date_str = job.get('end_date')
            if start_date_str and end_date_str:
                try:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                    if start_date > end_date:
                        reasons.append(f"Job at {job.get('company')} has start date {start_date_str} after end date {end_date_str}")
                except:
                    pass
                    
        years_of_exp = profile.get('years_of_experience', 0)
        total_career_months = sum(job.get('duration_months', 0) for job in career)
        total_career_years = total_career_months / 12.0
        if years_of_exp > total_career_years + 5.0 and len(career) > 0:
            reasons.append(f"Stated experience ({years_of_exp} years) is much larger than total career history ({total_career_years:.1f} years)")
            
        return len(reasons) > 0, reasons

    def _calculate_final_score(self, analysis: CandidateAnalysis) -> float:
        """
        Calculate weighted final score.
        """
        weights = {
            'skill_match': 0.25,         # 25% - Critical for role fit
            'behavioral': 0.25,         # 25% - Predicts actual availability
            'availability': 0.15,        # 15% - Immediate availability
            'experience': 0.12,          # 12% - Experience match
            'career': 0.10,              # 10% - Career progression
            'company': 0.08,             # 8% - Company quality
            'education': 0.03,            # 3% - Education
            'location': 0.02,             # 2% - Location preference
        }
        
        score = (
            analysis.skill_match_score * weights['skill_match'] +
            analysis.behavioral_signal_score * weights['behavioral'] +
            analysis.availability_score * weights['availability'] +
            analysis.experience_match_score * weights['experience'] +
            analysis.career_progression_score * weights['career'] +
            analysis.company_quality_score * weights['company'] +
            analysis.education_score * weights['education'] +
            analysis.location_score * weights['location']
        )
        
        return round(score, 4)
    
    def _generate_reasoning(self, analysis: CandidateAnalysis) -> str:
        """
        Generate human-readable reasoning for the ranking.
        """
        parts = []
        
        if analysis.skill_match_score >= 70:
            parts.append(f"Strong skill match ({analysis.skill_match_score:.0f}%)")
        elif analysis.skill_match_score >= 50:
            parts.append(f"Moderate skill match ({analysis.skill_match_score:.0f}%)")
        
        if analysis.behavioral_signal_score >= 60:
            parts.append("Active on platform with good engagement")
        elif analysis.behavioral_signal_score < 30:
            parts.append("Low platform engagement")
        
        if analysis.positive_signals:
            top_signal = analysis.positive_signals[0]
            parts.append(top_signal)
        
        if analysis.experience_match_score >= 90:
            parts.append("Experience level matches well")
        
        if analysis.availability_score >= 80:
            parts.append("Immediately available")
        elif analysis.availability_score < 50:
            parts.append("May not be immediately available")
        
        if analysis.company_quality_score >= 70:
            parts.append("Strong product company background")
        elif analysis.company_quality_score < 40:
            parts.append("Limited product company exposure")
            
        if getattr(analysis, 'is_hidden_gem', False):
            parts.append("Hidden Gem: Exceptional local talent")
            
        if getattr(analysis, 'commute_recommendation', ''):
            parts.append(f"Commute: {analysis.commute_recommendation}")
        
        if analysis.disqualifying_factors:
            main_issue = analysis.disqualifying_factors[0]
            parts.append(f"Concern: {main_issue}")
        
        return "; ".join(parts) if parts else "Average candidate"
        
    def rank_candidates(self, top_n: int = 100) -> List[Tuple[str, float, str]]:
        """
        Rank all candidates and return top N with scores and reasoning.
        """
        if not self.job_requirements:
            raise ValueError("Job requirements not set. Call parse_job_description first.")
        
        # 1. Compute semantic similarities for all candidates
        self.semantic_scores = self._compute_semantic_similarities(getattr(self, 'jd_text', ''))
        
        # 2. Find max similarity to normalize
        self.max_semantic_score = max(self.semantic_scores.values()) if self.semantic_scores else 1.0
        if self.max_semantic_score == 0:
            self.max_semantic_score = 1.0
            
        results = []
        for candidate in self.candidates:
            analysis = self.analyze_candidate(candidate)
            
            # Skip highly disqualified candidates or honeypots
            if len(analysis.disqualifying_factors) >= 2 or analysis.is_honeypot:
                continue
            
            results.append((
                analysis.candidate_id,
                analysis.final_score,
                analysis.reasoning
            ))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def get_detailed_analysis(self, candidate_id: str) -> Optional[CandidateAnalysis]:
        """Get detailed analysis for a specific candidate"""
        if not hasattr(self, 'semantic_scores'):
            self.semantic_scores = self._compute_semantic_similarities(getattr(self, 'jd_text', ''))
            self.max_semantic_score = max(self.semantic_scores.values()) if self.semantic_scores else 1.0
            if self.max_semantic_score == 0:
                self.max_semantic_score = 1.0
                
        for candidate in self.candidates:
            if candidate['candidate_id'] == candidate_id:
                return self.analyze_candidate(candidate)
        return None


def run_ranking(jd_text: str, candidates: List[Dict], 
                output_file: str = "submission.csv", top_n: int = 100):
    """
    Main function to run the ranking engine.
    """
    engine = RedrobIntelligenceEngine()
    
    # Parse job requirements
    engine.job_requirements = engine.parse_job_description(jd_text)
    
    # Load candidates
    engine.load_candidates(candidates)
    
    # Rank candidates
    rankings = engine.rank_candidates(top_n=top_n)
    
    # Write output
    with open(output_file, 'w') as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for rank, (cand_id, score, reasoning) in enumerate(rankings, 1):
            # Escape commas in reasoning
            reasoning_escaped = reasoning.replace(',', ';').replace('\n', ' ')
            f.write(f"{cand_id},{rank},{score:.4f},{reasoning_escaped}\n")
    
    print(f"Ranking complete. Top {len(rankings)} candidates written to {output_file}")
    
    return rankings


if __name__ == "__main__":
    # Load job description
    with open("job_description.txt", "r") as f:
        jd_text = f.read()
    
    # Load candidates (this would be from the actual data file)
    # For testing, we'll create sample data
    print("Redrob Intelligence Engine initialized")