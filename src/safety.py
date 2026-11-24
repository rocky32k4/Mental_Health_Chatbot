"""
safety.py
---------
Phase 5 : Safety Layer

All safety logic lives here. Imported by rag_chain.py and app.py.

Handles:
  1. Crisis detection       — suicidal/self-harm language
  2. Scope limiting         — out-of-bounds requests
  3. Disclaimer injection   — medication, diagnosis, legal topics
  4. Response validation    — catch unsafe bot responses before sending

Usage:
    from src.safety import check_input, check_response, get_disclaimer
"""


# ─────────────────────────────────────────────
# CRISIS KEYWORDS & RESPONSE
# ─────────────────────────────────────────────

# Grouped by severity for better detection
CRISIS_KEYWORDS = {

    "explicit_suicidal": [
        "kill myself", "killing myself",
        "end my life", "ending my life",
        "take my life", "taking my life",
        "want to die", "wanted to die",
        "wish i was dead", "wish i were dead",
        "better off dead", "better off without me",
        "don't want to live", "dont want to live",
        "no reason to live", "nothing to live for",
        "suicidal", "suicide",
    ],

    "self_harm": [
        "cut myself", "cutting myself",
        "hurt myself", "hurting myself",
        "harm myself", "harming myself",
        "self harm", "self-harm",
        "self injury", "self-injury",
        "burn myself", "burning myself",
        "overdose", "took too many pills",
        "took a lot of pills",
    ],

    "crisis_indicators": [
        "can't go on", "cannot go on",
        "can't take it anymore", "cannot take it anymore",
        "ending it", "ending it all",
        "i give up on life",
        "no point in living",
        "rather be dead",
        "don't care if i die", "dont care if i die",
        "goodbye forever",
        "final goodbye",
    ],
}

# Flatten all keywords into one list for quick lookup
ALL_CRISIS_KEYWORDS = [
    kw
    for group in CRISIS_KEYWORDS.values()
    for kw in group
]

CRISIS_RESPONSE = """I hear you, and I'm really glad you reached out.
What you're feeling sounds incredibly painful — you don't have to face this alone.

Please contact a crisis helpline right now. They are free, confidential, and available 24/7:

  iCall (India)          : 9152987821
  Vandrevala Foundation  : 1860-2662-345 (24/7)
  iCall WhatsApp         : +91 9152987821
  SNEHI                  : 044-24640050

If you are in immediate danger, please call 112 or go to your nearest hospital.

I'm here to listen. Would you like to talk about what's been going on?"""


# ─────────────────────────────────────────────
# OUT OF SCOPE TOPICS
# ─────────────────────────────────────────────

OUT_OF_SCOPE = {

    "medical_prescriptions": [
        "prescribe", "prescription",
        "what medication should i take",
        "which medicine", "what medicine",
        "what drug should",
        "dosage", "mg should i take",
    ],

    "diagnosis": [
        "do i have", "have i got",
        "am i depressed", "am i bipolar",
        "am i schizophrenic", "am i anxious",
        "diagnose me", "what disorder do i have",
        "what's wrong with me mentally",
        "what mental illness do i have",
    ],

    "legal_advice": [
        "is it legal", "am i liable",
        "can i sue", "legal advice",
        "what does the law say",
    ],

    "financial_advice": [
        "should i invest", "which stocks", "what stocks", "stock market",
        "financial advice", "money advice",
        "should i buy crypto",
    ],

    "unrelated_tasks": [
        "write my essay", "do my homework",
        "write my code", "fix my code",
        "translate this", "summarize this article",
        "what's the weather", "who won the match",
        "tell me a joke", "play a game",
    ],
}

OUT_OF_SCOPE_RESPONSE = """I'm specifically designed to provide mental health support and psychoeducation.
I'm not able to help with that particular request.

If you have questions about your mental health or emotional wellbeing,
I'm here to listen and support you. Is there something on your mind you'd like to talk about?"""


# ─────────────────────────────────────────────
# DISCLAIMER TRIGGERS
# ─────────────────────────────────────────────

# These topics get an automatic disclaimer appended to the response
DISCLAIMER_TRIGGERS = {

    "professional_help": [
        "therapy", "therapist", "counselling", "counseling",
        "psychiatrist", "psychologist", "mental health professional",
        "should i see", "should i go to",
    ],

    "medication": [
        "antidepressant", "medication", "medicine", "drug",
        "ssri", "snri", "benzodiazepine", "lithium",
        "prozac", "zoloft", "xanax", "valium",
    ],

    "diagnosis_adjacent": [
        "depression", "anxiety", "bipolar", "schizophrenia",
        "ocd", "ptsd", "adhd", "eating disorder",
        "borderline", "personality disorder",
    ],
}

DISCLAIMERS = {
    "professional_help": (
        "\n\n---\n"
        "💙 *Remember: I'm an AI assistant, not a licensed therapist. "
        "For professional support, please consider reaching out to a "
        "qualified mental health professional.*"
    ),
    "medication": (
        "\n\n---\n"
        "⚕️ *Note: Any information about medication is for educational purposes only. "
        "Always consult a licensed doctor or psychiatrist before starting, "
        "stopping, or changing any medication.*"
    ),
    "diagnosis_adjacent": (
        "\n\n---\n"
        "📋 *Note: I can provide general information about mental health conditions, "
        "but I cannot diagnose. Only a qualified mental health professional "
        "can provide a diagnosis.*"
    ),
}


# ─────────────────────────────────────────────
# UNSAFE RESPONSE PATTERNS
# ─────────────────────────────────────────────

# If the LLM accidentally produces these in its response, flag it
UNSAFE_RESPONSE_PATTERNS = [
    "you have depression",
    "you have anxiety",
    "you are bipolar",
    "you are schizophrenic",
    "you should take",
    "i recommend taking",
    "take this medication",
    "you need medication",
    "you are diagnosed",
    "i diagnose",
]


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def check_input(user_message: str) -> dict:
    """
    Analyse user input for safety concerns before sending to LLM.

    Returns:
        {
            "is_crisis"     : True/False,
            "is_out_of_scope: True/False,
            "crisis_type"   : which group triggered (or None),
            "override_response": response to send directly (or None),
        }

    If override_response is not None — send it directly, skip the LLM.
    """
    message_lower = user_message.lower().strip()

    # ── 1. Crisis check ──
    for group_name, keywords in CRISIS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message_lower:
                return {
                    "is_crisis"         : True,
                    "is_out_of_scope"   : False,
                    "crisis_type"       : group_name,
                    "override_response" : CRISIS_RESPONSE,
                }

    # ── 2. Out of scope check ──
    for group_name, keywords in OUT_OF_SCOPE.items():
        for keyword in keywords:
            if keyword in message_lower:
                return {
                    "is_crisis"         : False,
                    "is_out_of_scope"   : True,
                    "crisis_type"       : None,
                    "override_response" : OUT_OF_SCOPE_RESPONSE,
                }

    # ── All clear ──
    return {
        "is_crisis"         : False,
        "is_out_of_scope"   : False,
        "crisis_type"       : None,
        "override_response" : None,
    }


def check_response(bot_response: str) -> dict:
    """
    Validate the LLM's response before sending to user.
    Catches cases where the bot accidentally diagnoses or prescribes.

    Returns:
        {
            "is_safe"      : True/False,
            "flagged_phrase: the problematic phrase (or None),
            "safe_response": cleaned response or fallback,
        }
    """
    response_lower = bot_response.lower()

    for pattern in UNSAFE_RESPONSE_PATTERNS:
        if pattern in response_lower:
            return {
                "is_safe"       : False,
                "flagged_phrase": pattern,
                "safe_response" : (
                    "I want to be careful here — I'm not qualified to make "
                    "clinical assessments or recommendations. What I can do is "
                    "offer support and information.\n\n"
                    "Could you tell me more about what you're experiencing? "
                    "I'd like to understand how you're feeling."
                ),
            }

    return {
        "is_safe"       : True,
        "flagged_phrase": None,
        "safe_response" : bot_response,
    }


def get_disclaimer(user_message: str) -> str:
    """
    Check if a disclaimer should be appended to the response.
    Returns the disclaimer string, or empty string if none needed.

    Priority order: professional_help > medication > diagnosis_adjacent
    Only one disclaimer per response.
    """
    message_lower = user_message.lower()

    for trigger_group, keywords in DISCLAIMER_TRIGGERS.items():
        for keyword in keywords:
            if keyword in message_lower:
                return DISCLAIMERS[trigger_group]

    return ""


def process(user_message: str, bot_response: str) -> dict:
    """
    Full safety pipeline — call this in rag_chain.py after getting LLM response.

    Combines check_input + check_response + get_disclaimer in one call.

    Args:
        user_message : what the user typed
        bot_response : what the LLM returned

    Returns:
        {
            "final_response" : the safe, final response to send to user,
            "is_crisis"      : True if crisis was detected,
            "is_out_of_scope": True if out of scope,
            "disclaimer_added: True if a disclaimer was appended,
            "was_flagged"    : True if LLM response was unsafe and replaced,
        }
    """
    # Step 1 — Check input
    input_check = check_input(user_message)

    # If crisis or out of scope — return override immediately
    if input_check["override_response"]:
        return {
            "final_response"  : input_check["override_response"],
            "is_crisis"       : input_check["is_crisis"],
            "is_out_of_scope" : input_check["is_out_of_scope"],
            "disclaimer_added": False,
            "was_flagged"     : False,
        }

    # Step 2 — Validate LLM response
    response_check = check_response(bot_response)
    safe_response  = response_check["safe_response"]

    # Step 3 — Append disclaimer if needed
    disclaimer      = get_disclaimer(user_message)
    final_response  = safe_response + disclaimer
    disclaimer_added = bool(disclaimer)

    return {
        "final_response"  : final_response,
        "is_crisis"       : False,
        "is_out_of_scope" : False,
        "disclaimer_added": disclaimer_added,
        "was_flagged"     : not response_check["is_safe"],
    }


# ─────────────────────────────────────────────
# RUN DIRECTLY — TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  PHASE 5 — SAFETY LAYER TEST")
    print("=" * 55)

    # Test cases
    tests = [
        # (user_message, mock_bot_response, expected_behavior)
        (
            "I want to kill myself",
            "",
            "should trigger crisis response"
        ),
        (
            "I've been cutting myself",
            "",
            "should trigger crisis response"
        ),
        (
            "what stocks should I buy",
            "",
            "should be out of scope"
        ),
        (
            "I think I have depression",
            "Based on what you said, you have depression and should take antidepressants.",
            "should flag unsafe LLM response"
        ),
        (
            "how can therapy help me",
            "Therapy can help you develop coping strategies and understand your emotions better.",
            "should add professional help disclaimer"
        ),
        (
            "I feel really anxious today",
            "It sounds like you're having a tough day. Anxiety can feel overwhelming.",
            "should pass through cleanly"
        ),
    ]

    for i, (user_msg, bot_resp, expected) in enumerate(tests, 1):
        print(f"\nTest {i}: {expected}")
        print(f"  User    : {user_msg}")

        result = process(user_msg, bot_resp)

        print(f"  Crisis  : {result['is_crisis']}")
        print(f"  OOScope : {result['is_out_of_scope']}")
        print(f"  Flagged : {result['was_flagged']}")
        print(f"  Disclaimer: {result['disclaimer_added']}")
        print(f"  Response: {result['final_response'][:120]}...")

    print("\n" + "=" * 55)
    print("  Safety layer test complete")
    print("=" * 55)