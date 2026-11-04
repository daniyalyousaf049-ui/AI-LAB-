# Lab 6: NLP Fundamentals - Tokenization, Embeddings & Text Chunking
# Student: Daniyal Yousaf | Roll #: 097 | Section: BSAI-3B
# Course: CS4944 - Artificial Intelligence

import re
import json
import csv
import time
import sys

# avoid Windows console crashing on odd characters
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
except ImportError:
    print("Missing packages. Please run:")
    print("  pip install numpy scikit-learn sentence-transformers")
    raise SystemExit(1)


# ----- save everything printed to run_output.txt as well -----
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


OUTPUT_FILE = None


def start_logging():
    global OUTPUT_FILE
    OUTPUT_FILE = open("run_output.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, OUTPUT_FILE)


def stop_logging():
    global OUTPUT_FILE
    sys.stdout = sys.__stdout__
    if OUTPUT_FILE is not None:
        OUTPUT_FILE.close()
        OUTPUT_FILE = None


# ============================================================
# Retriever class (tfidf + embedding)
# ============================================================
class Retriever:
    """Simple search engine that can use TF-IDF or sentence embeddings."""

    def __init__(self, records, backend="tfidf", model=None):
        self.records = records
        self.backend = backend
        self.texts = []
        for r in records:
            self.texts.append(r["text"])

        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.doc_matrix = self.vectorizer.fit_transform(self.texts)
            self.model = None
            self.doc_embeddings = None
        else:
            # reuse a preloaded model when given (avoids reloading every config)
            if model is None:
                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
                )
            else:
                self.model = model
            self.doc_embeddings = self.model.encode(
                self.texts, normalize_embeddings=True, convert_to_numpy=True
            )
            self.vectorizer = None
            self.doc_matrix = None

    def search(self, query, k=3):
        """Return top-k (record, score) tuples for a query."""
        results = []
        if query.strip() == "" or k <= 0:
            return results

        if self.backend == "tfidf":
            q_vec = self.vectorizer.transform([query])
            if q_vec.nnz == 0:
                return results
            scores = (self.doc_matrix @ q_vec.T).toarray().flatten()
        else:
            q_emb = self.model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0]
            scores = np.dot(self.doc_embeddings, q_emb)

        scored = []
        for i in range(len(scores)):
            scored.append((i, float(scores[i])))

        scored.sort(key=lambda x: x[1], reverse=True)

        top = scored[:k]
        for idx, score in top:
            results.append((self.records[idx], score))

        return results


# ============================================================
# Chunking helpers (Task 2)
# ============================================================
def word_spans(text):
    """Return list of (start, end) character offsets for each non-whitespace word."""
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end()))
    return spans


def windows(text, size=40, overlap=10, offset=0):
    """Fixed word-window chunker. Returns list of (start_char, end_char) offsets."""
    # invalid size/overlap must raise, not silently fix the step
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")

    spans = word_spans(text)
    if len(spans) == 0:
        return []

    step = size - overlap
    chunks = []
    i = 0
    while i < len(spans):
        end_i = i + size
        if end_i > len(spans):
            end_i = len(spans)
        start_char = offset + spans[i][0]
        end_char = offset + spans[end_i - 1][1]
        chunks.append((start_char, end_char))
        if end_i >= len(spans):
            break
        i = i + step

    return chunks


def sentence_chunks(text, size=40):
    """
    Sentence-aware chunker.
    Split on sentence boundaries, then pack sentences into chunks of up to `size` words.
    A sentence longer than the budget is split with windows() as a fallback.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    # teaching splitter: punctuation followed by whitespace
    parts = re.split(r"([.!?]\s+)", text)
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.match(r"[.!?]\s+", parts[i + 1]):
            sentences.append(parts[i] + parts[i + 1])
            i = i + 2
        else:
            if parts[i].strip() != "":
                sentences.append(parts[i])
            i = i + 1

    # map each sentence back to character offsets in the original text
    sent_info = []
    search_from = 0
    for s in sentences:
        pos = text.find(s, search_from)
        if pos < 0:
            pos = search_from
        start = pos
        end = pos + len(s)
        while end > start and text[end - 1].isspace():
            end = end - 1
        word_count = len(re.findall(r"\S+", text[start:end]))
        sent_info.append({"start": start, "end": end, "words": word_count})
        search_from = pos + len(s)

    chunks = []
    cur_start = None
    cur_end = None
    cur_words = 0

    for info in sent_info:
        # long sentence: flush current chunk, then fallback-window the long one
        if info["words"] > size:
            if cur_start is not None:
                chunks.append((cur_start, cur_end))
                cur_start = None
                cur_end = None
                cur_words = 0
            segment = text[info["start"]:info["end"]]
            fallback = windows(segment, size=size, overlap=0, offset=info["start"])
            for span in fallback:
                chunks.append(span)
            continue

        if cur_start is None:
            cur_start = info["start"]
            cur_end = info["end"]
            cur_words = info["words"]
        elif cur_words + info["words"] <= size:
            cur_end = info["end"]
            cur_words = cur_words + info["words"]
        else:
            chunks.append((cur_start, cur_end))
            cur_start = info["start"]
            cur_end = info["end"]
            cur_words = info["words"]

    if cur_start is not None:
        chunks.append((cur_start, cur_end))

    return chunks


def make_chunks(docs, strategy):
    """
    Convert document dicts into chunk records.
    strategy: "0overlap", "10overlap", or "sentence"
    Each record has: id, doc, start, end, text
    """
    records = []

    for doc_name in docs:
        text = docs[doc_name]

        if strategy == "0overlap":
            spans = windows(text, size=40, overlap=0)
        elif strategy == "10overlap":
            spans = windows(text, size=40, overlap=10)
        else:
            spans = sentence_chunks(text, size=40)

        local_i = 0
        for start, end in spans:
            chunk_text = text[start:end]
            record = {
                "id": doc_name + ":" + str(local_i),
                "doc": doc_name,
                "start": start,
                "end": end,
                "text": chunk_text,
            }
            records.append(record)
            local_i = local_i + 1

    return records


# ============================================================
# Synthetic handbook documents (150-250 words each)
# Includes rule+exception, one >40-word sentence, and conflicts
# ============================================================
DOCS = {
    "late": (
        "University late submission policy applies to all undergraduate coursework in taught modules. "
        "Late work receives a 10 percent penalty for each calendar day late, up to three days, and is "
        "refused after that unless a formal extension was approved in writing before the deadline. "
        "Extensions may be granted for documented medical issues, family emergencies, or verified system "
        "outages, but students must email the module leader with a medical certificate before the due date. "
        "Late penalties are applied automatically by the grading system and appear on the feedback sheet. "
        "Group projects follow the same rules unless the brief states otherwise. Students who miss the "
        "absolute cutoff should contact student services to discuss deferred assessment options. Appeal "
        "requests must be filed within five working days of the grade release. Teaching assistants cannot "
        "waive late penalties without approval from the module leader. Keep copies of all submission "
        "receipts for your records. Weekend days still count as calendar days for the penalty clock. "
        "Resit coursework uses the same late rules unless the exam board publishes a different notice. "
        "Please note a conflicting portal notice elsewhere may mention a five day window, but this late "
        "policy document is the assessment office source for day based penalties on taught modules. "
        "Markers should never invent a local late rule that contradicts this assessment office text."
    ),
    "password": (
        "Password reset procedures for university accounts are handled through the IT self-service portal. "
        "Navigate to the login page and select Forgot Password, then enter your student email and complete "
        "an identity check with your student ID and date of birth. "
        "A reset link is sent to your registered personal email within ten minutes and the link expires "
        "after one hour. New passwords must be at least twelve characters and include upper case, lower "
        "case, a digit and a symbol, and reusing any of the last five passwords is not allowed. "
        "If you do not receive the email, check the spam folder first. Staff at the IT help desk can reset "
        "passwords in person after photo ID verification. Multi-factor authentication must be re-enrolled "
        "after a password change. Shared passwords between classmates violate the acceptable use policy "
        "and may lead to account suspension. Contact helpdesk@university.edu if the portal reports an "
        "error during reset. Temporary guest accounts cannot use the self-service reset form. Always "
        "sign out of shared lab machines after changing credentials so the old session cannot be reused. "
        "Store recovery codes offline and never send them by chat or email to anyone claiming to be IT support."
    ),
    "assignment": (
        "Assignment submission guidelines require all coursework to be uploaded through the virtual "
        "learning environment before the stated deadline. Accepted file formats are PDF for written "
        "reports and ZIP for programming projects. "
        "Filenames should follow the pattern ModuleCode_StudentID_Assignment so markers can match files "
        "to students without opening every archive by hand during busy marking weeks at the end of term "
        "when dozens of late uploads and resubmissions arrive in the same short window and support tickets "
        "already fill the help desk queue. "
        "The plagiarism checker runs automatically on every upload. Similarity scores above 25 percent "
        "are flagged for manual review by the module leader. Students may upload a draft once for "
        "feedback before the final submission. Only the last upload before the deadline is graded. "
        "Technical problems during upload should be reported immediately with screenshots to the help "
        "desk so a late penalty waiver can be considered. Printed hard copies are not accepted unless "
        "the brief specifically requests them. Group assignments need one designated member to submit "
        "on behalf of the team. Confirmation emails are sent after a successful upload. "
        "Important conflict note: the portal help page states late uploads are accepted for up to five "
        "calendar days with a flat five percent penalty applied once, which disagrees with the late "
        "policy day based ten percent rule, so always confirm with your module leader which source applies."
    ),
    "contractor": (
        "Contractor access policy covers external vendors working on university systems and buildings. "
        "Contractors receive temporary accounts limited to the systems listed in their statement of work. "
        "Access is reviewed every ninety days and removed when the contract ends. Contractors must "
        "complete security awareness training before credentials are issued. They cannot sponsor guest "
        "Wi-Fi for others and must not store university data on personal devices. Data export requires "
        "written approval from the information security office. Contractor accounts use multi-factor "
        "authentication and are monitored for unusual activity. Lost badges must be reported within "
        "twenty-four hours. Permanent staff accounts must never be shared with contractors. Requests "
        "for elevated privileges go through the change advisory board. Breach of these rules can "
        "terminate the contract and trigger a security incident review. "
        "Contractors can work remotely only if the signed contract explicitly permits remote work, and "
        "employee remote work rules do not apply to contractors even when both groups use the same VPN. "
        "The policy does not apply to full-time employees or enrolled students seeking temporary lab "
        "accounts, who must use the student account process instead of contractor forms."
    ),
    "printer": (
        "Campus printer and print quota rules are managed by IT services for students and visitors. "
        "Each student receives a monthly quota of 150 pages in black and white. Colour pages cost three "
        "times as much against the quota. Quotas reset on the first day of each month and unused pages "
        "do not roll over. Additional pages can be purchased through the student portal in packs of "
        "fifty. Printers are located in the library, lab buildings, and student centre. Jobs older than "
        "twenty-four hours in the print queue are deleted automatically. Paper jams should be reported "
        "using the help button on the printer panel. Do not attempt to open the printer casing yourself. "
        "Double-sided printing is the default to save paper. Large posters must use the dedicated "
        "plotter in the media lab and require staff booking. Guest printing is available with a prepaid "
        "card from the library desk. Misuse of printers for commercial work is prohibited under campus "
        "rules. For blank printer pages, check ink levels and run the cleaning utility, then print a "
        "test page afterwards before opening a hardware ticket. "
        "A related but conflicting notice in some lab doors still says the monthly student quota is 200 "
        "pages, but the IT services printer document here is the current source for the 150 page limit."
    ),
}


# labeled rule+exception strings used for integrity checks and hits
RULE_EXCEPTIONS = [
    {
        "doc": "late",
        "label": "late_penalty_with_extension",
        "evidence": (
            "Late work receives a 10 percent penalty for each calendar day late, up to three days, and is "
            "refused after that unless a formal extension was approved in writing before the deadline."
        ),
    },
    {
        "doc": "password",
        "label": "password_rules",
        "evidence": (
            "New passwords must be at least twelve characters and include upper case, lower "
            "case, a digit and a symbol, and reusing any of the last five passwords is not allowed."
        ),
    },
    {
        "doc": "contractor",
        "label": "contractor_remote",
        "evidence": (
            "Contractors can work remotely only if the signed contract explicitly permits remote work, and "
            "employee remote work rules do not apply to contractors even when both groups use the same VPN."
        ),
    },
]


# ============================================================
# Labeled queries (4 dev + 4 test answerable + 2 unanswerable)
# evidence = full answer including condition/exception
# ============================================================
DEV_QUERIES = [
    {
        "id": "dev1",
        "query": "What is the late penalty and when is late work refused?",
        "doc": "late",
        "evidence": (
            "Late work receives a 10 percent penalty for each calendar day late, up to three days, and is "
            "refused after that unless a formal extension was approved in writing before the deadline."
        ),
        "answerable": True,
        "split": "dev",
    },
    {
        "id": "dev2",
        "query": "How do I reset my university password?",
        "doc": "password",
        "evidence": (
            "Navigate to the login page and select Forgot Password, then enter your student email and complete "
            "an identity check with your student ID and date of birth."
        ),
        "answerable": True,
        "split": "dev",
    },
    {
        "id": "dev3",
        "query": "What file formats are accepted for assignment upload?",
        "doc": "assignment",
        "evidence": "Accepted file formats are PDF for written reports and ZIP for programming projects.",
        "answerable": True,
        "split": "dev",
    },
    {
        "id": "dev4",
        "query": "Can contractors work remotely?",
        "doc": "contractor",
        "evidence": (
            "Contractors can work remotely only if the signed contract explicitly permits remote work, and "
            "employee remote work rules do not apply to contractors even when both groups use the same VPN."
        ),
        "answerable": True,
        "split": "dev",
    },
]

TEST_QUERIES = [
    {
        "id": "test1",
        "query": "What proof is needed when I was ill and need more time?",
        "doc": "late",
        "evidence": (
            "Extensions may be granted for documented medical issues, family emergencies, or verified system "
            "outages, but students must email the module leader with a medical certificate before the due date."
        ),
        "answerable": True,
        "split": "test",
    },
    {
        "id": "test2",
        "query": "What are the rules for a new password?",
        "doc": "password",
        "evidence": (
            "New passwords must be at least twelve characters and include upper case, lower "
            "case, a digit and a symbol, and reusing any of the last five passwords is not allowed."
        ),
        "answerable": True,
        "split": "test",
    },
    {
        "id": "test3",
        "query": "How many pages is the monthly student print quota?",
        "doc": "printer",
        "evidence": "Each student receives a monthly quota of 150 pages in black and white.",
        "answerable": True,
        "split": "test",
    },
    {
        "id": "test4",
        "query": "What should I do for blank printer pages?",
        "doc": "printer",
        "evidence": (
            "For blank printer pages, check ink levels and run the cleaning utility, then print a "
            "test page afterwards before opening a hardware ticket."
        ),
        "answerable": True,
        "split": "test",
    },
    {
        "id": "unans1",
        "query": "What are tomorrow's flight prices?",
        "doc": None,
        "evidence": None,
        "answerable": False,
        "split": "test",
    },
    {
        "id": "unans2",
        "query": "Who won the football match?",
        "doc": None,
        "evidence": None,
        "answerable": False,
        "split": "test",
    },
]


# ============================================================
# FAQ data for Task 1 chatbot (includes 2 extra FAQs with sources)
# ============================================================
FAQ = [
    {
        "id": "faq:password",
        "question": "How do I reset my password?",
        "answer": "Use the recovery link on the login page.",
        "source": "it_help.txt",
    },
    {
        "id": "faq:submit",
        "question": "Where do I submit my assignment?",
        "answer": "Use the assignment upload page in the learning portal.",
        "source": "assessment.txt",
    },
    {
        "id": "faq:late",
        "question": "Can illness justify a late submission?",
        "answer": "Request an extension with medical evidence before the deadline.",
        "source": "handbook.txt",
    },
    # two new FAQs required by the lab
    {
        "id": "faq:printer",
        "question": "What is the monthly student print quota?",
        "answer": "Students get 150 black and white pages each month.",
        "source": "printer_policy.txt",
    },
    {
        "id": "faq:wifi",
        "question": "How do I connect to campus Wi-Fi?",
        "answer": "Join Eduroam and sign in with your university email and password.",
        "source": "network_guide.txt",
    },
]


def build_faq_bot(backend="tfidf"):
    records = []
    for r in FAQ:
        item = dict(r)
        item["text"] = r["question"] + " " + r["answer"]
        records.append(item)
    return Retriever(records, backend)


def faq_reply(engine, query):
    if not query.strip():
        return "Please enter a question."

    # exact canonical FAQ text gets the stored answer + source
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"

    hits = engine.search(query, k=3)
    if not hits:
        return "No matching FAQ. Please contact the help desk."

    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append(r["question"] + " (similarity " + f"{score:.3f}" + ")")
    return "\n".join(lines)


# ============================================================
# Evaluation helpers (Hit@3 / MRR@3 need doc id + full evidence)
# ============================================================
def check_hit(results, q):
    """Hit if top-3 chunk has matching doc id AND full evidence substring."""
    if q.get("doc") is None or q.get("evidence") is None:
        return False, 0.0

    rank = 1
    for record, score in results:
        if record["doc"] == q["doc"] and q["evidence"] in record["text"]:
            return True, 1.0 / rank
        rank = rank + 1
    return False, 0.0


def count_words(text):
    return len(re.findall(r"\S+", text))


def chunk_stats(records, docs):
    """Print count, min/max/avg words, duplication, rule+exception coverage."""
    if len(records) == 0:
        print("  No chunks.")
        return

    word_counts = []
    total_chunk_words = 0
    for r in records:
        n = count_words(r["text"])
        word_counts.append(n)
        total_chunk_words = total_chunk_words + n

    original_words = 0
    for name in docs:
        original_words = original_words + count_words(docs[name])

    duplicated = total_chunk_words - original_words
    avg_words = sum(word_counts) / len(word_counts)

    print("  chunk_count:", len(records))
    print("  min_words:", min(word_counts), "max_words:", max(word_counts),
          "avg_words:", round(avg_words, 2))
    print("  duplicated_words (chunk_total - original):", duplicated)

    print("  rule+exception intact in one chunk?")
    for item in RULE_EXCEPTIONS:
        found = False
        for r in records:
            if r["doc"] == item["doc"] and item["evidence"] in r["text"]:
                found = True
                break
        print("   ", item["label"], "->", found)


def audit_token_lengths(model, texts, label):
    lengths = []
    for t in texts:
        ids = model.tokenizer(t, truncation=False)["input_ids"]
        lengths.append(len(ids))
    max_len = max(lengths) if lengths else 0
    print("  Token audit (" + label + "): max_tokens=" + str(max_len),
          "limit=" + str(model.max_seq_length),
          "fits=" + str(max_len <= model.max_seq_length))
    if max_len > model.max_seq_length:
        raise ValueError("Model limit exceeded; rechunk or shorten query")
    return max_len


# ============================================================
# Task 1: Tokenization + FAQ chatbot demos
# ============================================================
def run_task1(model):
    print("=" * 60)
    print("TASK 1: Tokenization Comparison")
    print("=" * 60)
    print("MARKER_TOKEN_TABLE_START")

    # six inputs: punctuation, contraction, email, course id, empty string, dotted filename
    text_list = [
        "AI, search!",
        "I can't log in.",
        "help@example.com",
        "CS4944 version 2.1",
        "",
        "file_v2.1-final",
    ]

    print("")
    print(f"{'Input':<22} {'Whitespace':<40} {'Regex':<40} {'Model subwords (2 EN)'}")
    print("-" * 140)

    for idx, text in enumerate(text_list):
        ws_tokens = text.split()
        re_tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)

        ws_str = str(ws_tokens) + " n=" + str(len(ws_tokens))
        re_str = str(re_tokens) + " n=" + str(len(re_tokens))

        model_str = "-"
        # model tokenizer only for first two English examples
        if idx < 2:
            model_str = str(model.tokenizer.tokenize(text))

        display = text if text != "" else "(empty)"
        print(f"{display:<22} {ws_str:<40} {re_str:<40} {model_str}")

    print("MARKER_TOKEN_TABLE_END")
    print("")
    print("Notes:")
    print("- Whitespace keeps punctuation attached (e.g. 'AI,' and 'search!').")
    print("- Regex \\b\\w+\\b drops punctuation but splits emails and contractions like can't.")
    print("- Empty string gives zero tokens for both methods.")
    print("- Model subwords are different again; do not treat the regex as a full tokenizer.")
    print("")

    # FAQ chatbot demos (TF-IDF) for the report transcripts
    print("=" * 60)
    print("TASK 1: FAQ Chatbot (TF-IDF) demos")
    print("=" * 60)
    print("MARKER_FAQ_START")
    engine = build_faq_bot("tfidf")

    demo_queries = [
        "How do I reset my password?",
        "password recovery",
        "",
        "volcano geology",
        "What is the monthly student print quota?",
        "How do I connect to campus Wi-Fi?",
    ]
    for q in demo_queries:
        print("You:", repr(q) if q == "" else q)
        print("Bot:", faq_reply(engine, q))
        print("")
    print("MARKER_FAQ_END")
    print("Exact questions return the stored answer plus source.")
    print("Other questions only show suggestions so we confirm before treating them as answers.")
    print("")


def run_faq_chatbot_interactive():
    print("=" * 60)
    print("FAQ Chatbot interactive (TF-IDF). Type exit to stop.")
    print("=" * 60)
    engine = build_faq_bot("tfidf")
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting FAQ chatbot.")
            break
        if query.strip().lower() == "exit":
            break
        print("Bot:", faq_reply(engine, query))


# ============================================================
# Task 2: tests, stats, paraphrase retrieval
# ============================================================
def run_chunk_tests():
    print("=" * 60)
    print("TASK 2: Chunking tests")
    print("=" * 60)
    print("MARKER_CHUNK_TESTS_START")

    # empty text
    assert windows("") == []
    print("PASS: windows('') -> []")

    # document shorter than budget
    short = windows("short text")
    assert len(short) == 1
    print("PASS: short document -> one window", short)

    # invalid overlap
    try:
        windows("a b c d", size=2, overlap=2)
        print("FAIL: invalid overlap was accepted")
        raise AssertionError("Invalid overlap was accepted")
    except ValueError:
        print("PASS: overlap >= size raises ValueError")

    try:
        windows("a b c d", size=2, overlap=-1)
        raise AssertionError("negative overlap accepted")
    except ValueError:
        print("PASS: negative overlap raises ValueError")

    # sentence longer than budget
    long = " ".join(["word"] * 95) + "."
    long_spans = sentence_chunks(long, size=40)
    ok = True
    for a, b in long_spans:
        if len(long[a:b].split()) > 40:
            ok = False
    assert ok
    print("PASS: long sentence fallback, spans=", len(long_spans),
          "max_words=", max(len(long[a:b].split()) for a, b in long_spans))

    # offset integrity for make_chunks
    tiny = {"example": "Reset your password. Keep recovery codes private."}
    for strategy in ["0overlap", "10overlap", "sentence"]:
        for c in make_chunks(tiny, strategy):
            assert c["text"] == tiny[c["doc"]][c["start"]:c["end"]]
    print("PASS: make_chunks text == source[start:end]")
    print("MARKER_CHUNK_TESTS_END")
    print("")


def run_task2(model):
    run_chunk_tests()

    print("=" * 60)
    print("TASK 2: Chunk stats per strategy")
    print("=" * 60)
    print("MARKER_CHUNK_STATS_START")

    # show word counts for docs
    for name in DOCS:
        print("doc", name, "words=", count_words(DOCS[name]))

    strategies = ["0overlap", "10overlap", "sentence"]
    for strategy in strategies:
        print("")
        print("Strategy:", strategy)
        records = make_chunks(DOCS, strategy)
        chunk_stats(records, DOCS)

    print("MARKER_CHUNK_STATS_END")
    print("")
    print("Sentence splitter note: abbreviations like Mr. or e.g. can still break on the period.")
    print("")

    # three paraphrased queries, TF-IDF vs embeddings on sentence chunks
    print("=" * 60)
    print("TASK 2: Paraphrase retrieval (sentence chunks)")
    print("=" * 60)
    print("MARKER_PARAPHRASE_START")

    records = make_chunks(DOCS, "sentence")
    paraphrases = [
        "How can I regain access to my account?",
        "I was sick; can I hand work in tomorrow?",
        "My printer has power but the pages come out blank.",
    ]

    for backend in ["tfidf", "embedding"]:
        print("")
        print("Backend:", backend)
        if backend == "embedding":
            engine = Retriever(records, backend="embedding", model=model)
        else:
            engine = Retriever(records, backend="tfidf")
        for q in paraphrases:
            print("Query:", q)
            hits = engine.search(q, k=3)
            if not hits:
                print("  No matching records.")
            for record, score in hits:
                print(" ", record["id"], "score=" + f"{score:.3f}",
                      "doc=" + record["doc"],
                      "offsets=(" + str(record["start"]) + "-" + str(record["end"]) + ")")
                preview = record["text"]
                if len(preview) > 160:
                    preview = preview[:160] + "..."
                print("   evidence:", preview)

    print("MARKER_PARAPHRASE_END")
    print("")


# ============================================================
# Task 3: retrieval evaluation + document chatbot
# ============================================================
def run_task3(model):
    print("=" * 60)
    print("TASK 3: Retrieval evaluation (dev/test, 6 configs)")
    print("=" * 60)

    all_queries = DEV_QUERIES + TEST_QUERIES
    query_texts = []
    for q in all_queries:
        query_texts.append(q["query"])

    strategies = ["0overlap", "10overlap", "sentence"]
    backends = ["tfidf", "embedding"]

    metrics_rows = []
    logs = []
    failures = []

    print("MARKER_METRICS_START")
    print(f"{'Strategy':<12} {'Backend':<10} {'Split':<6} {'Hit@3':>7} {'MRR@3':>7} "
          f"{'Chunks':>7} {'Encode(s)':>9} {'Search(s)':>9}")
    print("-" * 80)

    for strategy in strategies:
        records = make_chunks(DOCS, strategy)
        texts = []
        for r in records:
            texts.append(r["text"])

        # audit token lengths once per strategy
        print("Token audit for strategy", strategy)
        audit_token_lengths(model, texts + query_texts, strategy)

        for backend in backends:
            t0 = time.perf_counter()
            if backend == "tfidf":
                vectorizer = TfidfVectorizer()
                C = vectorizer.fit_transform(texts)
                Q = vectorizer.transform(query_texts)
                encode_seconds = time.perf_counter() - t0
                t1 = time.perf_counter()
                score_matrix = (Q @ C.T).toarray()
                search_seconds = time.perf_counter() - t1
            else:
                C = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
                Q = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
                encode_seconds = time.perf_counter() - t0
                t1 = time.perf_counter()
                score_matrix = Q @ C.T
                search_seconds = time.perf_counter() - t1

            # separate metrics for dev and test
            for split_name in ["dev", "test"]:
                hits_list = []
                rr_list = []
                for qi in range(len(all_queries)):
                    q = all_queries[qi]
                    if q["split"] != split_name:
                        continue
                    if not q["answerable"]:
                        continue
                    row = score_matrix[qi]
                    # stable argsort descending
                    ranked = list(np.argsort(-row, kind="stable")[:3])
                    good_ranks = []
                    rank = 1
                    for idx in ranked:
                        rec = records[idx]
                        if rec["doc"] == q["doc"] and q["evidence"] in rec["text"]:
                            good_ranks.append(rank)
                        rank = rank + 1
                    hits_list.append(1 if len(good_ranks) > 0 else 0)
                    if len(good_ranks) > 0:
                        rr_list.append(1.0 / good_ranks[0])
                    else:
                        rr_list.append(0.0)

                if len(hits_list) == 0:
                    hit_at_3 = 0.0
                    mrr_at_3 = 0.0
                else:
                    hit_at_3 = sum(hits_list) / len(hits_list)
                    mrr_at_3 = sum(rr_list) / len(rr_list)

                metrics_rows.append({
                    "strategy": strategy,
                    "backend": backend,
                    "split": split_name,
                    "hit_at_3": hit_at_3,
                    "mrr_at_3": mrr_at_3,
                    "n_chunks": len(records),
                    "encode_sec": encode_seconds,
                    "search_sec": search_seconds,
                })
                print(
                    f"{strategy:<12} {backend:<10} {split_name:<6} "
                    f"{hit_at_3:>7.3f} {mrr_at_3:>7.3f} {len(records):>7} "
                    f"{encode_seconds:>9.3f} {search_seconds:>9.3f}"
                )

            # print top-3 for every query and collect failures
            print("")
            print("Top-3 details:", strategy, backend)
            for qi in range(len(all_queries)):
                q = all_queries[qi]
                row = score_matrix[qi]
                ranked = list(np.argsort(-row, kind="stable")[:3])
                print("Query:", q["query"], "| split=", q["split"],
                      "| answerable=", q["answerable"])
                rank = 1
                hit_this = False
                first_rec = None
                first_rank = None
                for idx in ranked:
                    rec = records[idx]
                    score = float(row[idx])
                    print(
                        "  rank", rank,
                        "id=" + rec["id"],
                        "score=" + f"{score:.4f}",
                        "doc=" + rec["doc"],
                        "offsets=(" + str(rec["start"]) + "-" + str(rec["end"]) + ")"
                    )
                    preview = rec["text"]
                    if len(preview) > 140:
                        preview = preview[:140] + "..."
                    print("   text:", preview)

                    logs.append({
                        "strategy": strategy,
                        "method": backend,
                        "query": q["query"],
                        "split": q["split"],
                        "rank": rank,
                        "score": score,
                        "chunk_id": rec["id"],
                        "doc": rec["doc"],
                        "start": rec["start"],
                        "end": rec["end"],
                        "text": rec["text"],
                        "answerable": q["answerable"],
                    })

                    if first_rec is None:
                        first_rec = rec
                        first_rank = rank

                    if q["answerable"] and rec["doc"] == q["doc"] and q["evidence"] in rec["text"]:
                        hit_this = True
                    rank = rank + 1

                if q["answerable"] and not hit_this:
                    failures.append({
                        "strategy": strategy,
                        "backend": backend,
                        "query": q["query"],
                        "expected_doc": q["doc"],
                        "rank": first_rank,
                        "retrieved_id": first_rec["id"] if first_rec else None,
                        "retrieved_doc": first_rec["doc"] if first_rec else None,
                        "retrieved_text": first_rec["text"] if first_rec else None,
                    })
            print("")

    print("MARKER_METRICS_END")

    # choose strategy from DEV results only (prefer higher Hit@3, then MRR@3)
    print("")
    print("MARKER_SELECTION_START")
    best = None
    for row in metrics_rows:
        if row["split"] != "dev":
            continue
        if best is None:
            best = row
            continue
        if row["hit_at_3"] > best["hit_at_3"]:
            best = row
        elif row["hit_at_3"] == best["hit_at_3"] and row["mrr_at_3"] > best["mrr_at_3"]:
            best = row
        elif (row["hit_at_3"] == best["hit_at_3"] and row["mrr_at_3"] == best["mrr_at_3"]
              and row["backend"] == "embedding" and best["backend"] != "embedding"):
            # prefer embedding when tied so paraphrase chatbot works better
            best = row

    selected_strategy = best["strategy"]
    selected_backend = best["backend"]
    print("Selected from DEV only:", selected_strategy, "+", selected_backend,
          "Hit@3=" + f"{best['hit_at_3']:.3f}", "MRR@3=" + f"{best['mrr_at_3']:.3f}")

    # report matching TEST metrics without retuning
    for row in metrics_rows:
        if row["strategy"] == selected_strategy and row["backend"] == selected_backend and row["split"] == "test":
            print("Held-out TEST for selected config: Hit@3=" + f"{row['hit_at_3']:.3f}",
                  "MRR@3=" + f"{row['mrr_at_3']:.3f}")
    print("MARKER_SELECTION_END")

    # print at least 3 real failures
    print("")
    print("MARKER_FAILURES_START")
    print("Automatic failure list (answerable misses in top-3):")
    shown = 0
    for f in failures:
        print("FAIL", shown + 1)
        print("  config:", f["strategy"], f["backend"])
        print("  query:", f["query"])
        print("  expected_doc:", f["expected_doc"])
        print("  retrieved_rank:", f["rank"], "id=", f["retrieved_id"], "doc=", f["retrieved_doc"])
        preview = f["retrieved_text"] or ""
        if len(preview) > 180:
            preview = preview[:180] + "..."
        print("  retrieved_text:", preview)
        shown = shown + 1
        if shown >= 8:
            break
    if shown == 0:
        print("No answerable misses found in this run.")
    print("MARKER_FAILURES_END")

    # save csv log
    if len(logs) > 0:
        with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            writer.writeheader()
            writer.writerows(logs)
        print("Wrote retrieval_log.csv rows=", len(logs))

    with open("lab6_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics_rows, "selected": best, "failures": failures[:8]}, f, indent=2)

    # scripted document chatbot demos for the report
    print("")
    print("=" * 60)
    print("TASK 3: Document chatbot demos (selected config)")
    print("=" * 60)
    print("MARKER_DOC_CHAT_START")
    records = make_chunks(DOCS, selected_strategy)
    if selected_backend == "embedding":
        search_engine = Retriever(records, backend="embedding", model=model)
    else:
        search_engine = Retriever(records, backend="tfidf")

    demo_cases = [
        ("exact-ish", "What is the late penalty and when is late work refused?", "1"),
        ("paraphrase", "How can I recover my login access?", "1"),
        ("empty", "", ""),
        ("unsupported", "What are tomorrow's flight prices?", ""),
    ]
    for label, question, choice in demo_cases:
        print("DEMO:", label)
        print("Question:", repr(question) if question == "" else question)
        if question.strip() == "":
            print("No candidate evidence; rephrase or contact a human.")
            print("")
            continue
        hits = search_engine.search(question, k=3)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            print("")
            continue
        print("Candidate evidence; relevance must be checked:")
        for number in range(len(hits)):
            r, score = hits[number]
            print(number + 1, r["id"], f"{score:.3f}", "characters", r["start"], r["end"])
            preview = r["text"]
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print("   ", preview)
        # unanswerable: never auto-assert an answer
        if label == "unsupported" or choice == "":
            print("No answer selected. Rephrase or contact a human.")
        else:
            idx = int(choice) - 1
            r, score = hits[idx]
            print("Retrieved passage:", r["text"])
            print("Source:", r["doc"], "characters", r["start"], r["end"])
        print("")

    print("MARKER_DOC_CHAT_END")
    print("")
    return selected_strategy, selected_backend


def run_chatbot(model, strategy, backend):
    print("=" * 60)
    print("TASK 3 Chatbot: Document Q&A (" + strategy + " + " + backend + ")")
    print("=" * 60)
    print("Type a question about university policies.")
    print("Type 'exit' to stop.")
    print("Candidates are evidence to inspect, not verified answers.")
    print("")

    records = make_chunks(DOCS, strategy)
    if backend == "embedding":
        search_engine = Retriever(records, backend="embedding", model=model)
    else:
        search_engine = Retriever(records, backend=backend)

    while True:
        try:
            question = input("Question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot.")
            break

        if question.strip().lower() == "exit":
            print("Goodbye.")
            break

        if question.strip() == "":
            print("No candidate evidence; rephrase or contact a human.")
            continue

        hits = search_engine.search(question, k=3)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            continue

        print("Candidate evidence; relevance must be checked:")
        for number in range(len(hits)):
            record, score = hits[number]
            print(number + 1, record["id"], f"{score:.3f}",
                  "characters", record["start"], record["end"])
            preview = record["text"]
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print("   ", preview)

        chosen = input("Choose 1-3 to inspect; Enter if none apply: ").strip()
        if not chosen.isdigit() or not (1 <= int(chosen) <= len(hits)):
            print("No answer selected. Rephrase or contact a human.")
            continue

        record, score = hits[int(chosen) - 1]
        print("Retrieved passage:", record["text"])
        print("Source:", record["doc"], "characters", record["start"], record["end"])
        print("")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    start_logging()
    try:
        print("Lab 6 run")
        print("Python:", sys.version.replace("\n", " "))
        print("numpy:", np.__version__)
        print("scikit-learn:", sklearn.__version__)
        print("sentence-transformers:", sentence_transformers.__version__)
        print("MARKER_VERSIONS_START")
        print("python=" + sys.version.split()[0])
        print("numpy=" + np.__version__)
        print("scikit-learn=" + sklearn.__version__)
        print("sentence-transformers=" + sentence_transformers.__version__)
        print("MARKER_VERSIONS_END")
        print("")

        # load model once, time it separately
        print("MARKER_TIMING_START")
        t_load = time.perf_counter()
        shared_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        model_load_sec = time.perf_counter() - t_load
        print("Model load seconds:", round(model_load_sec, 3))
        print("Model max_seq_length:", shared_model.max_seq_length)
        print("MARKER_TIMING_END")
        print("")

        run_task1(shared_model)
        run_task2(shared_model)
        selected_strategy, selected_backend = run_task3(shared_model)

        # interactive loops only when a real terminal is attached
        if sys.stdin.isatty():
            print("Opening FAQ chatbot next (type exit when finished).")
            run_faq_chatbot_interactive()
            print("Opening document chatbot next (type exit when finished).")
            run_chatbot(shared_model, selected_strategy, selected_backend)
        else:
            print("Skipping interactive chatbots (no TTY). Scripted demos are already in run_output.txt.")
        print("Done. Full transcript saved to run_output.txt")
    finally:
        stop_logging()
