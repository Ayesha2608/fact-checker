"""Command-line interface for the fact-checking chatbot."""

from database import DatabaseManager
from nlp_evaluation import NLPEvaluator
from reporting import ReportGenerator
from search_engine import has_network
from utils import configure_logging
from verifier import FactCheckingEngine


class FactCheckingCLI:
    """Clean menu-driven CLI."""

    def __init__(self):
        configure_logging()
        self.engine = FactCheckingEngine()
        self.database = DatabaseManager()
        self.reporter = ReportGenerator()
        self.evaluator = NLPEvaluator()
        self.last_fact_check_id = None
        self.last_result = None

    def run(self):
        """Start the interactive application."""
        print("AI-Powered Evidence Analysis and Fact Verification Platform")
        print("Python Standard Library Only | Live Retrieval Diagnostics Enabled")
        if not has_network():
            print("Warning: network check failed. Live search may not work on this connection.")
        while True:
            print("\nMenu")
            print("1. Verify Claim / Ask Question")
            print("2. Verify News")
            print("3. View History")
            print("4. Search History")
            print("5. Feedback")
            print("6. Analytics Dashboard")
            print("7. Export History CSV")
            print("8. Export Latest Full Report")
            print("9. NLP Pipeline Demo / Offline Evaluation")
            print("10. Exit")
            choice = input("Choose an option: ").strip()
            if choice == "1":
                self.ask_question()
            elif choice == "2":
                self.verify_news()
            elif choice == "3":
                self.view_history()
            elif choice == "4":
                self.search_history()
            elif choice == "5":
                self.feedback()
            elif choice == "6":
                self.analytics()
            elif choice == "7":
                self.export_history()
            elif choice == "8":
                self.export_latest_report()
            elif choice == "9":
                self.nlp_demo()
            elif choice == "10":
                print("Goodbye.")
                break
            else:
                print("Invalid option. Please choose 1-10.")

    def ask_question(self):
        query = input("Enter your question or claim: ").strip()
        if not query:
            print("Please enter a question or claim.")
            return
        result = self.engine.verify_claim(query)
        self.last_fact_check_id = result.get("fact_check_id")
        self.last_result = result
        print(self.reporter.text_report(result))

    def verify_news(self):
        headline = input("Paste news headline: ").strip()
        if not headline:
            print("Please enter a headline.")
            return
        result = self.engine.verify_claim(headline, mode="news")
        self.last_fact_check_id = result.get("fact_check_id")
        self.last_result = result
        print(self.reporter.text_report(result))

    def view_history(self):
        rows = self.database.list_history()
        if not rows:
            print("No history yet.")
            return
        for row in rows:
            confidence = "N/A" if row[6] is None else str(row[6]) + "%"
            print("#{0} | {2} | {3} | {4} | {5} | {6} | quality={7}% | {1}".format(*row[:6], confidence, row[7]))

    def search_history(self):
        keyword = input("Search keyword: ").strip()
        rows = self.database.search_history(keyword)
        if not rows:
            print("No matching history records.")
            return
        for row in rows:
            confidence = "N/A" if row[6] is None else str(row[6]) + "%"
            print("#{0} | {2} | {3} | {4} | {5} | {6} | quality={7}% | {1}".format(*row[:6], confidence, row[7]))

    def feedback(self):
        if not self.last_fact_check_id:
            value = input("Enter fact-check ID for feedback: ").strip()
            if not value.isdigit():
                print("A numeric fact-check ID is required.")
                return
            fact_check_id = int(value)
        else:
            fact_check_id = self.last_fact_check_id
            print("Using latest fact-check ID:", fact_check_id)
        helpful = input("Was this helpful? (y/n): ").strip().lower().startswith("y")
        comment = input("Optional comment: ").strip()
        self.database.add_feedback(fact_check_id, helpful, comment)
        stats = self.database.feedback_statistics()
        print("Feedback saved. Helpful: {0}, Not helpful: {1}".format(stats["helpful"], stats["not_helpful"]))

    def export_history(self):
        rows = self.database.list_history(limit=1000)
        path = self.reporter.export_history_csv(rows, "history_export.csv")
        print("Exported:", path)

    def analytics(self):
        summary = self.database.analytics_summary()
        print(self.reporter.analytics_report(summary))

    def export_latest_report(self):
        if not getattr(self, "last_result", None):
            print("No verification report is available in this session yet.")
            return
        claim_id = self.last_result.get("claim_id", "latest").lower()
        path = self.reporter.export_full_report(self.last_result, claim_id + "_verification_report.txt")
        print("Exported:", path)

    def nlp_demo(self):
        print("\nNLP Demo")
        print("1. Show pipeline for custom text")
        print("2. Run offline evaluation corpus")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            text = input("Enter text: ").strip()
            if not text:
                print("Please enter text.")
                return
            result = self.evaluator.demo_pipeline(text)
            summary = result.get("pipeline_summary", {})
            print("Stages:", summary.get("stages", []))
            print("Normalized:", summary.get("normalization", ""))
            print("Tokens:", result.get("tokens", []))
            print("Lemmas:", summary.get("lemmas", []))
            print("Bigrams:", summary.get("bigrams", []))
            print("POS Sample:", summary.get("pos_sample", []))
            print("Entities:", summary.get("entities", {}))
            print("Noun Phrases:", summary.get("noun_phrases", []))
            print("Verb Phrases:", summary.get("verb_phrases", []))
            print("Claim Structure:", summary.get("claim_structure", {}))
            print("Lexical Diversity:", summary.get("lexical_diversity", 0), "%")
            print("Lexical Density:", summary.get("lexical_density", 0), "%")
        elif choice == "2":
            print(self.reporter.nlp_evaluation_report(self.evaluator.evaluate()))
        else:
            print("Invalid option.")


if __name__ == "__main__":
    FactCheckingCLI().run()
