#!/usr/bin/env python3

# cli utilities dependencies
import argparse
import random
import time
from pathlib import Path
from typing import Optional

# data visualization dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ui
from rich.console import Console
from rich.progress import track
from rich.table import Table
from pydantic import BaseModel, Field

console = Console()


class CompanyInput(BaseModel):
    name: str = Field(..., min_length=1)
    domain: str = Field(..., min_length=1)


class EnrichedCompany(CompanyInput):
    headcount: Optional[int] = None
    linkedin_url: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    founded_year: Optional[int] = None
    enrichment_success: bool = False


class ScoredCompany(EnrichedCompany):
    lead_score: float
    lead_grade: str
    headcount_tier: str
    fit_reason: str  # Why this company is a good/bad fit


class MockEnrichmentAPI:
    """
    API mock to simulate fetching enrichment data for a given company.
    """

    # Retrieved from the `Financial Services` category in https://www.linkedin.com/pulse/full-linkedin-industry-list-2024-upamanyu-roy-krihf
    INDUSTRIES = [
        # High-fit financial services
        "Banking",
        "Investment Banking",
        "Capital Markets",
        "Investment Management",
        "Venture Capital and Private Equity Principals",
        "Investment Advice",
        "Securities and Commodity Exchanges",
        "Funds and Trusts",
        # Medium-fit financial services
        "Credit Intermediation",
        "Insurance",
        "Insurance Carriers",
        "Insurance and Employee Benefit Funds",
        "Insurance Agencies and Brokerages",
        "Loan Brokers",
        "Savings Institutions",
        "Pension Funds",
        "Trusts and Estates",
        "Claims Adjusting, Actuarial Services",
        # Lower-fit (but still finance-related)
        "International Trade and Development",
        "Fundraising",
    ]

    # focusing on Europe as the main target market.
    LOCATIONS = [
        "London, UK",
        "Berlin, Germany",
        "Amsterdam, Netherlands",
        "Paris, France",
        "Frankfurt, Germany",
        "Munich, Germany",
        "Dublin, Ireland",
        "Stockholm, Sweden",
        "Copenhagen, Denmark",
        "Helsinki, Finland",
        "Vienna, Austria",
        "Zurich, Switzerland",
        "Luxembourg",
        "Brussels, Belgium",
        "Warsaw, Poland",
        "Madrid, Spain",
        "Barcelona, Spain",
        "Milan, Italy",
        "Lisbon, Portugal",
        "Prague, Czech Republic",
        "Bucharest, Romania",
        "Athens, Greece",
        "Budapest, Hungary",
        "Sofia, Bulgaria",
        "Zagreb, Croatia",
        "Tallinn, Estonia",
        "Riga, Latvia",
        "Vilnius, Lithuania",
        "Rotterdam, Netherlands",
        "Hamburg, Germany",
        "Lyon, France",
        "Oslo, Norway",
    ]

    HEADCOUNT_RANGES = [
        (50, 150),  # Small fintech
        (151, 300),  # Growing
        (301, 600),  # Medium
        (601, 1200),  # Large
        (1201, 3000),  # Enterprise
        (3001, 10000),  # Very Large
    ]

    def __init__(self, delay_range: tuple = (0.1, 0.5)):
        self.delay_range = delay_range

    def enrich_company(self, domain: str) -> Optional[dict]:
        time.sleep(random.uniform(*self.delay_range))

        headcount_range = random.choice(self.HEADCOUNT_RANGES)
        headcount = random.randint(*headcount_range)

        return {
            "headcount": headcount,
            "linkedin_url": f"https://www.linkedin.com/company/{domain.split('.')[0]}",
            "location": random.choice(self.LOCATIONS),
            "industry": random.choice(self.INDUSTRIES),
            "founded_year": random.randint(2010, 2025),
        }


class LeadScorer:
    @staticmethod
    def get_headcount_tier(headcount: Optional[int]) -> str:
        """Determine company size tier"""
        if headcount is None:
            return "unknown"
        if headcount < 50:
            return "startup"
        elif headcount <= 200:
            return "small"
        elif headcount <= 500:
            return "medium"
        elif headcount <= 1000:
            return "large"
        elif headcount <= 5000:
            return "enterprise"
        return "very_large"

    @staticmethod
    def calculate_headcount_score(headcount: Optional[int]) -> float:
        """
        Scores a company based on its headcount. Values go from 0 to 100.
        The optimal ICP is a large company with headcount around 1000 employees.
        """
        if headcount is None:
            return 50

        if headcount < 50:
            return 60
        elif headcount <= 200:
            return 75
        elif headcount <= 500:
            return 90
        elif headcount <= 1000:
            return 100  # Large (ICP)
        elif headcount <= 5000:
            return 85  # Enterprise
        else:
            return 70  # Very large

    @staticmethod
    def calculate_industry_score(industry: Optional[str]) -> float:
        """
        Scores a company based on the industry it operaates in. Values go from 0 to 100.
        """
        if industry is None:
            return 50  # Unknown industry

        # High-fit: Modern banking, investment, and capital markets
        high_fit = {
            "Banking",
            "Investment Banking",
            "Capital Markets",
            "Investment Management",
            "Venture Capital and Private Equity Principals",
            "Investment Advice",
            "Securities and Commodity Exchanges",
        }

        # Medium-fit: Traditional financial services
        medium_fit = {
            "Funds and Trusts",
            "Credit Intermediation",
            "Savings Institutions",
            "Loan Brokers",
        }

        # Lower-fit: Insurance and other finance-adjacent
        lower_fit = {
            "Insurance",
            "Insurance Carriers",
            "Insurance and Employee Benefit Funds",
            "Insurance Agencies and Brokerages",
            "Pension Funds",
            "Trusts and Estates",
            "Claims Adjusting, Actuarial Services",
            "International Trade and Development",
            "Fundraising",
        }

        if industry in high_fit:
            return 100  # perfect fit
        elif industry in medium_fit:
            return 70  # good fit
        elif industry in lower_fit:
            return 40  # lower fit
        else:
            return 30  # not a financial institution

    @classmethod
    def calculate_lead_score(cls, enriched: EnrichedCompany) -> float:
        """
        Calculates lead score (0-100).
        Formula: (Size + Industry) / 2
        """
        size_score = cls.calculate_headcount_score(enriched.headcount)
        industry_score = cls.calculate_industry_score(enriched.industry)

        total_score = (size_score + industry_score) / 2

        return round(total_score, 2)

    @staticmethod
    def assign_grade(score: float) -> str:
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"

    @staticmethod
    def generate_fit_reason(enriched: EnrichedCompany) -> str:
        """Generates a simple comment that describes why the score of a company."""
        size = enriched.headcount if enriched.headcount else "Unknown"
        industry = enriched.industry if enriched.industry else "Unknown"
        return f"Size: {size} employees, Industry: {industry}"

    @classmethod
    def score_company(cls, enriched: EnrichedCompany) -> ScoredCompany:
        """Scores an enriched company"""
        lead_score = cls.calculate_lead_score(enriched)
        lead_grade = cls.assign_grade(lead_score)
        headcount_tier = cls.get_headcount_tier(enriched.headcount)
        fit_reason = cls.generate_fit_reason(enriched)

        return ScoredCompany(
            **enriched.model_dump(),
            lead_score=lead_score,
            lead_grade=lead_grade,
            headcount_tier=headcount_tier,
            fit_reason=fit_reason,
        )


class CompanyEnricher:
    def __init__(self, api: MockEnrichmentAPI):
        self.api = api
        self.scorer = LeadScorer()

    def enrich_company(self, company_input: CompanyInput) -> EnrichedCompany:
        """Enrich a single company"""
        enrichment_data = self.api.enrich_company(company_input.domain)

        if enrichment_data:
            return EnrichedCompany(
                name=company_input.name,
                domain=company_input.domain,
                enrichment_success=True,
                **enrichment_data,
            )
        else:
            return EnrichedCompany(
                name=company_input.name,
                domain=company_input.domain,
                enrichment_success=False,
            )

    def enrich_batch(self, companies: list[CompanyInput]) -> list[EnrichedCompany]:
        """Enrich a batch of companies with progress tracking"""
        enriched = []

        for company in track(companies, description="[cyan]Enriching companies..."):
            enriched_company = self.enrich_company(company)
            enriched.append(enriched_company)

        return enriched

    def score_companies(self, enriched: list[EnrichedCompany]) -> list[ScoredCompany]:
        """Score all enriched companies"""
        return [self.scorer.score_company(company) for company in enriched]


class FunnelVisualizer:
    """
    Creates a visualization of the funnel using Seaborn and Matplotlib.
    """

    @staticmethod
    def create_funnel_visualization(
        scored_companies: list[ScoredCompany],
        output_path: Path = Path("finperks_lead_funnel.png"),
    ):
        """Creates comprehensive funnel."""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Lead Scoring Funnel & Analytics",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )

        # 1. Lead Quality Funnel
        ax1 = axes[0, 0]
        total = len(scored_companies)
        enriched = [c for c in scored_companies if c.enrichment_success]
        enriched_count = len(enriched)
        a_grade_leads = len([c for c in enriched if c.lead_grade == "A"])

        funnel_data = {
            "Total Companies": total,
            "Successfully Enriched": enriched_count,
            "A Grade Leads": a_grade_leads,
        }

        stages = list(funnel_data.keys())
        values = list(funnel_data.values())
        colors = ["#6366f1", "#8b5cf6", "#ec4899"]

        bars = ax1.barh(stages, values, color=colors, alpha=0.8, edgecolor="black")
        ax1.set_xlabel("Number of Companies", fontweight="bold")
        ax1.set_title("Lead Quality Funnel", fontweight="bold", fontsize=14)

        for i, (bar, value) in enumerate(zip(bars, values)):
            percentage = (value / total) * 100
            ax1.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value} ({percentage:.1f}%)",
                va="center",
                fontweight="bold",
            )

        ax1.set_xlim(0, total * 1.15)

        # 2. Score Distribution
        ax2 = axes[0, 1]
        scores = [c.lead_score for c in enriched]

        if scores:
            ax2.hist(scores, bins=20, color="#8b5cf6", alpha=0.7, edgecolor="black")
            ax2.axvline(
                sum(scores) / len(scores),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {sum(scores) / len(scores):.1f}",
            )
            ax2.set_xlabel("Lead Score", fontweight="bold")
            ax2.set_ylabel("Number of Companies", fontweight="bold")
            ax2.set_title("Lead Score Distribution", fontweight="bold", fontsize=14)
            ax2.legend()

        # 3. Grade Distribution
        ax3 = axes[0, 2]
        grades = [c.lead_grade for c in enriched]

        if grades:
            grade_counts = (
                pd.Series(grades)
                .value_counts()
                .reindex(["A", "B", "C", "D"], fill_value=0)
            )
            colors_grade = {
                "A": "#10b981",
                "B": "#3b82f6",
                "C": "#f59e0b",
                "D": "#ef4444",
            }
            bar_colors = [
                colors_grade.get(grade, "#95a5a6") for grade in grade_counts.index
            ]

            bars = ax3.bar(
                grade_counts.index,
                grade_counts.values,
                color=bar_colors,
                alpha=0.8,
                edgecolor="black",
            )
            ax3.set_xlabel("Lead Grade", fontweight="bold")
            ax3.set_ylabel("Number of Companies", fontweight="bold")
            ax3.set_title("Lead Grade Distribution", fontweight="bold", fontsize=14)

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        # 4. Industry Breakdown
        ax4 = axes[1, 0]
        industries = [c.industry for c in enriched if c.industry]

        if industries:
            industry_counts = pd.Series(industries).value_counts().head(8)
            bars = ax4.barh(
                industry_counts.index,
                industry_counts.values,
                color="#6366f1",
                alpha=0.8,
                edgecolor="black",
            )
            ax4.set_xlabel("Number of Companies", fontweight="bold")
            ax4.set_title("Industries (Fintech Focus)", fontweight="bold", fontsize=14)
            ax4.invert_yaxis()

            for bar in bars:
                width = bar.get_width()
                ax4.text(
                    width + 0.1,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{int(width)}",
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

        # 5. Geographic Distribution (EU Focus)
        ax5 = axes[1, 1]
        locations = [c.location for c in enriched if c.location]

        if locations:
            location_counts = pd.Series(locations).value_counts().head(8)
            colors_pie = plt.cm.Set3.colors[: len(location_counts)]

            wedges, texts, autotexts = ax5.pie(
                location_counts.values,
                labels=location_counts.index,
                autopct="%1.1f%%",
                colors=colors_pie,
                startangle=90,
                textprops={"fontsize": 8},
            )

            for autotext in autotexts:
                autotext.set_color("black")
                autotext.set_fontweight("bold")

            ax5.set_title(
                "Geographic Distribution (EU Markets)", fontweight="bold", fontsize=14
            )

        # 6. Company Size Distribution
        ax6 = axes[1, 2]
        tiers = [c.headcount_tier for c in enriched if c.headcount_tier != "unknown"]

        if tiers:
            tier_counts = pd.Series(tiers).value_counts()
            tier_order = [
                "too_small",
                "small",
                "growing",
                "medium",
                "large",
                "enterprise",
                "very_large",
            ]
            tier_counts = tier_counts.reindex(
                [t for t in tier_order if t in tier_counts.index]
            )

            # Highlight ideal sizes
            colors_tier = [
                "#ef4444"
                if t in ["too_small"]
                else "#10b981"
                if t in ["medium", "large", "enterprise"]
                else "#f59e0b"
                for t in tier_counts.index
            ]

            bars = ax6.bar(
                range(len(tier_counts)),
                tier_counts.values,
                color=colors_tier,
                alpha=0.8,
                edgecolor="black",
            )
            ax6.set_xticks(range(len(tier_counts)))
            ax6.set_xticklabels(tier_counts.index, rotation=45, ha="right")
            ax6.set_ylabel("Number of Companies", fontweight="bold")
            ax6.set_title(
                "Company Size (Green = Ideal Customer Persona)",
                fontweight="bold",
                fontsize=14,
            )

            for bar in bars:
                height = bar.get_height()
                ax6.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(
            f"\n[green]✓[/green] Visualization saved to [cyan]{output_path}[/cyan]"
        )
        plt.close()


def display_summary(scored_companies: list[ScoredCompany]):
    total = len(scored_companies)
    enriched = [c for c in scored_companies if c.enrichment_success]
    enriched_count = len(enriched)

    scores = [c.lead_score for c in enriched]
    avg_score = sum(scores) / len(scores) if scores else 0

    grade_counts = {}
    for c in enriched:
        grade_counts[c.lead_grade] = grade_counts.get(c.lead_grade, 0) + 1

    # Summary table
    table = Table(
        title="\n🎯 Lead Scoring Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Value", style="green", width=25)

    table.add_row("Total Companies Analyzed", str(total))
    table.add_row(
        "Successfully Enriched",
        f"{enriched_count} ({enriched_count / total * 100:.1f}%)",
    )
    table.add_row("Average Lead Score", f"{avg_score:.2f}")

    for grade in ["A", "B", "C", "D"]:
        count = grade_counts.get(grade, 0)
        if count > 0:
            style = (
                "bold green" if grade == "A" else "yellow" if grade == "B" else "dim"
            )
            table.add_row(f"Grade {grade} Leads", str(count), style=style)

    console.print(table)

    top_leads = sorted(enriched, key=lambda x: x.lead_score, reverse=True)[:10]

    if top_leads:
        console.print("\n🏆 [bold yellow]Top 10 Leads:[/bold yellow]")
        leads_table = Table(show_header=True, header_style="bold yellow")
        leads_table.add_column("Company", style="cyan", width=20)
        leads_table.add_column("Score", style="green", width=8)
        leads_table.add_column("Grade", style="magenta", width=8)
        leads_table.add_column("Size", style="blue", width=12)
        leads_table.add_column("Fit Reason", style="white", width=50)

        for lead in top_leads:
            leads_table.add_row(
                lead.name,
                f"{lead.lead_score:.1f}",
                lead.lead_grade,
                f"{lead.headcount}" if lead.headcount else "Unknown",
                lead.fit_reason,
            )

        console.print(leads_table)


def main():
    parser = argparse.ArgumentParser(
        description="Lead Finder - Find ideal  prospects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finperks_lead_finder.py fintech_prospects.csv
  python finperks_lead_finder.py banks.csv -o scored_leads.csv
  python finperks_lead_finder.py prospects.csv --no-viz
        """,
    )

    parser.add_argument(
        "input_file", type=Path, help="Input CSV file with columns: Name, Domain"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path (default: <input_name>_scored.csv)",
    )

    parser.add_argument(
        "--no-viz", action="store_true", help="Skip generating funnel visualization"
    )

    args = parser.parse_args()

    # Derive output file names from input file if not specified
    if args.output is None:
        input_stem = args.input_file.stem  # e.g., "fintech_prospects"
        args.output = args.input_file.parent / f"{input_stem}_scored.csv"

    # Derive visualization path from input file
    viz_output = args.input_file.parent / f"{args.input_file.stem}_funnel.png"

    # Display header
    console.print("\n[bold blue]🎯 Lead Finder[/bold blue]", justify="center")
    console.print(
        "[dim]Finding ideal bank & fintech prospects for cashback API[/dim]\n",
        justify="center",
    )

    # Read input CSV
    try:
        df = pd.read_csv(args.input_file)
        console.print(
            f"[green]✓[/green] Loaded {len(df)} companies from [cyan]{args.input_file}[/cyan]"
        )
    except FileNotFoundError:
        console.print(f"[red]✗[/red] File not found: {args.input_file}")
        return
    except Exception as e:
        console.print(f"[red]✗[/red] Error reading CSV: {e}")
        return

    required_columns = {"Name", "Domain"}
    if not required_columns.issubset(df.columns):
        console.print(f"[red]✗[/red] CSV must contain columns: {required_columns}")
        return

    try:
        companies = [
            CompanyInput(name=row["Name"], domain=row["Domain"])
            for _, row in df.iterrows()
        ]
    except Exception as e:
        console.print(f"[red]✗[/red] Error parsing company data: {e}")
        return

    api = MockEnrichmentAPI()
    enricher = CompanyEnricher(api)

    console.print()
    enriched_companies = enricher.enrich_batch(companies)

    console.print("[cyan]Calculating lead scores...[/cyan]")
    scored_companies = enricher.score_companies(enriched_companies)

    output_data = [company.model_dump() for company in scored_companies]
    output_df = pd.DataFrame(output_data)

    try:
        output_df.to_csv(args.output, index=False)
        console.print(
            f"[green]✓[/green] Scored leads saved to [cyan]{args.output}[/cyan]"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Error saving output: {e}")
        return

    display_summary(scored_companies)

    if not args.no_viz:
        console.print("\n[cyan]Generating lead funnel visualization...[/cyan]")
        try:
            FunnelVisualizer.create_funnel_visualization(scored_companies, viz_output)
        except Exception as e:
            console.print(
                f"[yellow]⚠[/yellow] Warning: Could not generate visualization: {e}"
            )

    console.print("\n[bold green]✨ Lead scoring complete![/bold green]\n")


if __name__ == "__main__":
    main()
