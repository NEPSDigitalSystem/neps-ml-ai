"""
NEPS Digital — Mock REDCap API Service
======================================
Simulates REDCap API responses for local development.
Teams can develop against this and swap to real REDCap by changing the base URL.

Usage:
    from app.services.redcap_mock import RedCapMockClient
    client = RedCapMockClient()
    participants = client.get_participants()

"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid


class SurveyStatus(str, Enum):
    COMPLETE = "2"
    UNVERIFIED = "1"
    INCOMPLETE = "0"


class ConsentStatus(str, Enum):
    CONSENTED = "consented"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"
    ASSENT_ONLY = "assent_only"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RedCapMockClient:
    """
    Mock REDCap API client that simulates all endpoints NEPS teams need.
    Generates realistic longitudinal data for Ghana, Sierra Leone, and Tanzania.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._participants = self._generate_participants()
        self._survey_responses = self._generate_survey_responses()
        self._consent_records = self._generate_consent_records()
        self._distress_screenings = self._generate_distress_screenings()
        self._referrals = []
        self._wp6_sessions = self._generate_wp6_sessions()

    # ─── PARTICIPANT REGISTRY ─────────────────────────────────────────

    def _generate_participants(self, count: int = 150) -> List[Dict]:
        """Generate realistic participant cohort across 3 countries."""
        countries = {
            "Ghana": ["Kumasi", "Accra", "Ho", "Tamale"],
            "Sierra Leone": ["Freetown", "Bo", "Makeni"],
            "Tanzania": ["Dar es Salaam", "Mwanza", "Arusha"]
        }

        schools = {
            "Kumasi": ["KNUST SHS", "Prempeh College", "Opoku Ware SHS"],
            "Accra": ["Accra Academy", "Achimota School", "Presbyterian Boys"],
            "Ho": ["Mawuli School", "OLA SHS", "Sogakope SHS"],
            "Tamale": ["Tamale SHS", "Ghana SHS", "St. Charles"],
            "Freetown": ["Prince of Wales", "Methodist Boys", "Annie Walsh"],
            "Bo": ["Bo Government", "Christ the King", "St. Francis"],
            "Makeni": ["Makeni Comprehensive", "Bombali", "St. Joseph"],
            "Dar es Salaam": ["Ilboru", "Tambaza", "Kisutu"],
            "Mwanza": ["Mwanza Academy", "St. Augustine", "VETA"],
            "Arusha": ["Arusha Secondary", "Korogwe", "Moshi"]
        }

        participants = []
        for i in range(1, count + 1):
            country = random.choice(list(countries.keys()))
            site = random.choice(countries[country])
            school = random.choice(schools.get(site, ["Unknown School"]))

            # Generate DOB for ages 12-24
            age = random.randint(12, 24)
            dob = datetime.now() - timedelta(days=age*365 + random.randint(0, 364))

            participant = {
                "record_id": f"NEPS-{country[:3].upper()}-{i:04d}",
                "redcap_event_name": "baseline_arm_1",
                "country": country,
                "site": site,
                "school": school,
                "age": age,
                "date_of_birth": dob.strftime("%Y-%m-%d"),
                "gender": random.choice(["Male", "Female", "Other", "Prefer not to say"]),
                "grade_level": random.randint(7, 12) if age <= 18 else random.choice(["University", "Vocational", "Not in school"]),
                "enrollment_date": (datetime.now() - timedelta(days=random.randint(30, 180))).strftime("%Y-%m-%d"),
                "cohort_status": random.choice(["active", "active", "active", "active", "inactive", "withdrawn"]),
                "phone_contact": f"+233{random.randint(200000000, 599999999)}" if country == "Ghana" else
                                f"+232{random.randint(30000000, 79999999)}" if country == "Sierra Leone" else
                                f"+255{random.randint(600000000, 799999999)}",
                "consent_status": random.choice([ConsentStatus.CONSENTED, ConsentStatus.CONSENTED,
                                                 ConsentStatus.CONSENTED, ConsentStatus.PENDING]),
                "redcap_data_access_group": site.lower().replace(" ", "_"),
                "redcap_repeat_instrument": "",
                "redcap_repeat_instance": "",
            }
            participants.append(participant)

        return participants

    def get_participants(self, country: Optional[str] = None,
                        site: Optional[str] = None,
                        status: Optional[str] = None) -> List[Dict]:
        """Get participants with optional filtering."""
        results = self._participants.copy()

        if country:
            results = [p for p in results if p["country"] == country]
        if site:
            results = [p for p in results if p["site"] == site]
        if status:
            results = [p for p in results if p["cohort_status"] == status]

        return results

    def get_participant(self, record_id: str) -> Optional[Dict]:
        """Get single participant by ID."""
        return next((p for p in self._participants if p["record_id"] == record_id), None)

    # ─── SURVEY RESPONSES ─────────────────────────────────────────────

    def _generate_survey_responses(self) -> Dict[str, List[Dict]]:
        """Generate monthly self-report and comprehensive wave data."""
        responses = {}

        for participant in self._participants:
            pid = participant["record_id"]
            participant_responses = []

            # Baseline (month 0)
            baseline = self._generate_monthly_response(pid, 0, is_baseline=True)
            participant_responses.append(baseline)

            # Monthly reports (1-24)
            for month in range(1, 25):
                if random.random() > 0.15:  # 85% completion rate
                    monthly = self._generate_monthly_response(pid, month)
                    participant_responses.append(monthly)

            # Comprehensive waves (6, 12, 18, 24)
            for wave_month in [6, 12, 18, 24]:
                if any(r.get("month") == wave_month for r in participant_responses):
                    wave = self._generate_comprehensive_wave(pid, wave_month)
                    participant_responses.append(wave)

            responses[pid] = participant_responses

        return responses

    def _generate_monthly_response(self, record_id: str, month: int,
                                   is_baseline: bool = False) -> Dict:
        """Generate monthly psychosocial monitoring data."""
        base_stress = random.uniform(15, 35)
        trend = month * random.uniform(-0.3, 0.5)  # Slight deterioration over time

        response = {
            "record_id": record_id,
            "redcap_event_name": f"month_{month}_arm_1" if not is_baseline else "baseline_arm_1",
            "month": month,
            "survey_date": (datetime.now() - timedelta(days=30*(24-month))).strftime("%Y-%m-%d"),
            "survey_complete": SurveyStatus.COMPLETE,

            # Core WP4 psychosocial indicators
            "perceived_stress_score": round(min(40, max(0, base_stress + trend + random.uniform(-5, 5))), 1),
            "mood_status": random.choice(["Good", "Fair", "Poor", "Very poor"]),
            "anxiety_score": round(random.uniform(0, 21), 1),
            "depression_score": round(random.uniform(0, 27), 1),
            "sleep_quality": random.choice(["Excellent", "Good", "Fair", "Poor"]),
            "daily_functioning": round(random.uniform(0, 100), 1),
            "fatigue_level": random.choice(["None", "Mild", "Moderate", "Severe"]),

            # Educational
            "school_attendance_days": random.randint(15, 22),
            "social_isolation_score": round(random.uniform(0, 10), 1),
            "coping_behaviours": random.choice(["Active", "Avoidant", "Social", "Substance use"]),
            "substance_use": random.choice(["None", "Alcohol", "Cannabis", "Other"]),

            # Safeguarding screening
            "suicidality_screening": random.choice(["No", "Passive thoughts", "Active plan", "Recent attempt"]),
            "self_esteem_score": round(random.uniform(10, 40), 1),
            "loneliness_score": round(random.uniform(0, 20), 1),

            # REDCap metadata
            "redcap_repeat_instrument": "monthly_self_report",
            "redcap_repeat_instance": str(month),
        }

        # Add risk flag if scores are high
        if response["anxiety_score"] > 15 or response["depression_score"] > 20 or response["suicidality_screening"] in ["Active plan", "Recent attempt"]:
            response["risk_flag"] = "HIGH"
            response["requires_follow_up"] = "1"
        else:
            response["risk_flag"] = "LOW"
            response["requires_follow_up"] = "0"

        return response

    def _generate_comprehensive_wave(self, record_id: str, month: int) -> Dict:
        """Generate comprehensive survey wave data (6m, 12m, 18m, 24m)."""
        return {
            "record_id": record_id,
            "redcap_event_name": f"month_{month}_arm_1",
            "wave_type": f"{month}_month",
            "survey_complete": SurveyStatus.COMPLETE,

            # Educational pressure
            "examination_stress": round(random.uniform(0, 10), 1),
            "academic_pressure": round(random.uniform(0, 10), 1),
            "homework_burden": round(random.uniform(0, 10), 1),
            "school_climate": random.choice(["Supportive", "Neutral", "Hostile"]),
            "bullying_exposure": random.choice(["None", "Verbal", "Physical", "Cyber", "Multiple"]),
            "harsh_discipline": random.choice(["Never", "Rarely", "Sometimes", "Often"]),
            "educational_aspirations": random.choice(["University", "Vocational", "Employment", "Undecided"]),
            "fear_of_failure": round(random.uniform(0, 10), 1),
            "teacher_support": round(random.uniform(0, 10), 1),
            "counselling_access": random.choice(["Yes", "No", "Don't know"]),

            # Poverty/socioeconomic
            "household_assets": random.randint(0, 20),
            "food_insecurity": random.choice(["None", "Mild", "Moderate", "Severe"]),
            "economic_strain": round(random.uniform(0, 10), 1),
            "employment_pressure": random.choice(["None", "Family expects", "Self pressure", "Financial need"]),
            "financial_stress": round(random.uniform(0, 10), 1),
            "digital_access": random.choice(["Smartphone", "Basic phone", "Shared", "None"]),
            "household_instability": random.choice(["Stable", "Some instability", "Highly unstable"]),

            # Mental health literacy & stigma
            "internalised_stigma": round(random.uniform(0, 10), 1),
            "community_stigma": round(random.uniform(0, 10), 1),
            "family_stigma": round(random.uniform(0, 10), 1),
            "school_stigma": round(random.uniform(0, 10), 1),
            "mental_health_literacy": round(random.uniform(0, 20), 1),
            "help_seeking_intention": random.choice(["Yes", "Maybe", "No"]),
            "help_seeking_behaviour": random.choice(["Professional", "Informal", "Religious", "None"]),
            "awareness_of_services": random.choice(["Good", "Some", "None"]),

            # Protective factors
            "resilience_score": round(random.uniform(0, 100), 1),
            "social_support": round(random.uniform(0, 20), 1),
            "family_connectedness": round(random.uniform(0, 20), 1),
            "peer_support": round(random.uniform(0, 20), 1),
            "community_connectedness": round(random.uniform(0, 20), 1),
            "religious_support": round(random.uniform(0, 20), 1),
            "school_belonging": round(random.uniform(0, 20), 1),

            "redcap_repeat_instrument": "comprehensive_wave",
            "redcap_repeat_instance": str(month // 6),
        }

    def get_survey_responses(self, record_id: Optional[str] = None,
                            event: Optional[str] = None,
                            instrument: Optional[str] = None) -> List[Dict]:
        """Get survey responses with filtering."""
        if record_id:
            responses = self._survey_responses.get(record_id, [])
        else:
            responses = [r for responses in self._survey_responses.values() for r in responses]

        if event:
            responses = [r for r in responses if r.get("redcap_event_name") == event]
        if instrument:
            responses = [r for r in responses if r.get("redcap_repeat_instrument") == instrument]

        return responses

    # ─── CONSENT MANAGEMENT ───────────────────────────────────────────

    def _generate_consent_records(self) -> List[Dict]:
        """Generate consent/assent tracking records."""
        records = []
        for p in self._participants:
            records.append({
                "record_id": p["record_id"],
                "consent_date": p["enrollment_date"],
                "consent_version": "v1.0",
                "consent_status": p["consent_status"],
                "guardian_consent": random.choice(["Yes", "Yes", "Yes", "N/A (18+)"]),
                "assent_status": random.choice(["Yes", "Yes", "Yes", "Pending"]),
                "consent_withdrawn": "0",
                "withdrawal_reason": "",
                "re_consent_required": random.choice(["0", "0", "0", "1"]),
                "re_consent_date": "",
            })
        return records

    def get_consent_status(self, record_id: str) -> Optional[Dict]:
        """Get consent status for a participant."""
        return next((c for c in self._consent_records if c["record_id"] == record_id), None)

    def update_consent(self, record_id: str, status: str) -> Dict:
        """Update consent status (simulates REDCap record update)."""
        record = self.get_consent_status(record_id)
        if record:
            record["consent_status"] = status
            record["re_consent_date"] = datetime.now().strftime("%Y-%m-%d")
        return record or {"error": "Record not found"}

    # ─── DISTRESS SCREENING & SAFEGUARDING ────────────────────────────

    def _generate_distress_screenings(self) -> List[Dict]:
        """Generate distress screening records with some high-risk flags."""
        screenings = []
        for p in self._participants:
            if random.random() > 0.9:  # 10% have distress flags
                screening = {
                    "record_id": p["record_id"],
                    "screening_date": datetime.now().strftime("%Y-%m-%d"),
                    "distress_score": round(random.uniform(15, 30), 1),
                    "suicidality_flag": random.choice(["1", "1", "0"]),
                    "severity": random.choice([RiskLevel.HIGH, RiskLevel.CRITICAL]),
                    "trigger_form": "monthly_self_report",
                    "trigger_item": "suicidality_screening",
                    "assigned_responder": random.choice(["Dr. Otu-Ansah", "Counselor A", "Counselor B"]),
                    "action_taken": "",
                    "referral_made": "0",
                    "referral_destination": "",
                    "welfare_check_due": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "resolution_status": "open",
                }
                screenings.append(screening)
        return screenings

    def get_distress_screenings(self, status: Optional[str] = None) -> List[Dict]:
        """Get distress screenings, optionally filtered by status."""
        if status:
            return [s for s in self._distress_screenings if s["resolution_status"] == status]
        return self._distress_screenings

    def create_referral(self, record_id: str, destination: str,
                       notes: str = "") -> Dict:
        """Create a safeguarding referral."""
        referral = {
            "referral_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
            "record_id": record_id,
            "initiation_date": datetime.now().strftime("%Y-%m-%d"),
            "destination": destination,
            "status": "initiated",
            "notes": notes,
            "follow_up_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
        }
        self._referrals.append(referral)
        return referral

    # ─── WP6 INTERVENTION TRACKING ────────────────────────────────────

    def _generate_wp6_sessions(self) -> Dict[str, List[Dict]]:
        """Generate WP6 intervention session data."""
        sessions = {}
        for p in self._participants[:20]:  # Only first 20 get interventions
            pid = p["record_id"]
            participant_sessions = []
            for session_num in range(1, 9):
                session = {
                    "record_id": pid,
                    "session_number": session_num,
                    "session_date": (datetime.now() - timedelta(days=30*(8-session_num))).strftime("%Y-%m-%d"),
                    "attendance": random.choice(["Present", "Present", "Present", "Absent", "Partial"]),
                    "engagement_level": round(random.uniform(1, 5), 1),
                    "fidelity_score": round(random.uniform(70, 100), 1),
                    "satisfaction_score": round(random.uniform(3, 5), 1),
                    "homework_completion": random.choice(["Complete", "Partial", "None"]),
                    "distress_pre": round(random.uniform(10, 25), 1),
                    "distress_post": round(random.uniform(5, 15), 1),
                }
                participant_sessions.append(session)
            sessions[pid] = participant_sessions
        return sessions

    def get_wp6_sessions(self, record_id: str) -> List[Dict]:
        """Get WP6 intervention sessions for a participant."""
        return self._wp6_sessions.get(record_id, [])

    # ─── DATA EXPORT (REDCap API format) ──────────────────────────────

    def export_records(self, format: str = "json",
                      fields: Optional[List[str]] = None,
                      events: Optional[List[str]] = None) -> Any:
        """Export records in REDCap API format."""
        all_records = []
        for pid, responses in self._survey_responses.items():
            for response in responses:
                if events and response.get("redcap_event_name") not in events:
                    continue
                if fields:
                    filtered = {k: v for k, v in response.items() if k in fields}
                    all_records.append(filtered)
                else:
                    all_records.append(response)

        if format == "csv":
            # Simple CSV conversion
            import csv
            import io
            output = io.StringIO()
            if all_records:
                # Union of all keys across all record types (monthly + comprehensive waves differ)
                all_keys = list(dict.fromkeys(k for r in all_records for k in r.keys()))
                writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_records)
            return output.getvalue()

        return all_records

    def export_metadata(self) -> Dict:
        """Export REDCap project metadata (data dictionary)."""
        return {
            "project_id": "NEPS-2025",
            "project_title": "NEPS Digital - Youth Mental Health Observatory",
            "creation_time": "2025-01-15",
            "production_time": "2025-03-01",
            "in_production": True,
            "purpose": "Research",
            "events": [
                {"event_name": "baseline_arm_1", "arm_num": 1, "day_offset": 0},
                {"event_name": "month_1_arm_1", "arm_num": 1, "day_offset": 30},
                {"event_name": "month_6_arm_1", "arm_num": 1, "day_offset": 180},
                {"event_name": "month_12_arm_1", "arm_num": 1, "day_offset": 365},
                {"event_name": "month_18_arm_1", "arm_num": 1, "day_offset": 545},
                {"event_name": "month_24_arm_1", "arm_num": 1, "day_offset": 730},
            ],
            "instruments": [
                {"instrument_name": "demographics", "instrument_label": "Demographics"},
                {"instrument_name": "monthly_self_report", "instrument_label": "Monthly Self-Report"},
                {"instrument_name": "comprehensive_wave", "instrument_label": "Comprehensive Survey Wave"},
                {"instrument_name": "distress_screening", "instrument_label": "Distress Screening"},
                {"instrument_name": "referral_form", "instrument_label": "Referral Form"},
                {"instrument_name": "wp6_session", "instrument_label": "WP6 Session Record"},
            ]
        }

    # ─── UTILITY ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get project statistics."""
        return {
            "total_participants": len(self._participants),
            "by_country": {
                "Ghana": len([p for p in self._participants if p["country"] == "Ghana"]),
                "Sierra Leone": len([p for p in self._participants if p["country"] == "Sierra Leone"]),
                "Tanzania": len([p for p in self._participants if p["country"] == "Tanzania"]),
            },
            "active_cohort": len([p for p in self._participants if p["cohort_status"] == "active"]),
            "total_surveys": sum(len(r) for r in self._survey_responses.values()),
            "high_risk_flags": len(self._distress_screenings),
            "open_referrals": len([r for r in self._referrals if r["status"] == "initiated"]),
            "wp6_enrolled": len(self._wp6_sessions),
        }
