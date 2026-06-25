"""
NEPS ML/AI — Unit Tests for RedCapMockClient
=============================================
Tests the mock REDCap API service used for local development and CI.
"""

import pytest
from app.services.redcap_mock import RedCapMockClient, RiskLevel, ConsentStatus, SurveyStatus


@pytest.fixture(scope="module")
def client():
    """Shared client instance for all tests (deterministic seed=42)."""
    return RedCapMockClient(seed=42)


# ─── PARTICIPANT REGISTRY ──────────────────────────────────────────────────────

class TestParticipants:
    def test_generates_150_participants(self, client):
        participants = client.get_participants()
        assert len(participants) == 150

    def test_participant_has_required_fields(self, client):
        p = client.get_participants()[0]
        required_fields = ["record_id", "country", "site", "age", "gender",
                           "consent_status", "cohort_status", "enrollment_date"]
        for field in required_fields:
            assert field in p, f"Missing field: {field}"

    def test_record_id_format(self, client):
        participants = client.get_participants()
        for p in participants:
            assert p["record_id"].startswith("NEPS-"), f"Bad ID: {p['record_id']}"

    def test_ages_are_in_range(self, client):
        participants = client.get_participants()
        for p in participants:
            assert 12 <= p["age"] <= 24, f"Age out of range: {p['age']}"

    def test_countries_are_valid(self, client):
        valid_countries = {"Ghana", "Sierra Leone", "Tanzania"}
        participants = client.get_participants()
        for p in participants:
            assert p["country"] in valid_countries, f"Unknown country: {p['country']}"

    def test_filter_by_country_ghana(self, client):
        ghana = client.get_participants(country="Ghana")
        assert len(ghana) > 0
        assert all(p["country"] == "Ghana" for p in ghana)

    def test_filter_by_country_sierra_leone(self, client):
        sle = client.get_participants(country="Sierra Leone")
        assert len(sle) > 0
        assert all(p["country"] == "Sierra Leone" for p in sle)

    def test_filter_by_country_tanzania(self, client):
        tz = client.get_participants(country="Tanzania")
        assert len(tz) > 0
        assert all(p["country"] == "Tanzania" for p in tz)

    def test_filter_by_active_status(self, client):
        active = client.get_participants(status="active")
        assert len(active) > 0
        assert all(p["cohort_status"] == "active" for p in active)

    def test_get_participant_by_id_returns_correct_record(self, client):
        all_participants = client.get_participants()
        first_id = all_participants[0]["record_id"]
        result = client.get_participant(first_id)
        assert result is not None
        assert result["record_id"] == first_id

    def test_get_participant_by_invalid_id_returns_none(self, client):
        result = client.get_participant("NEPS-DOES-NOT-EXIST-9999")
        assert result is None

    def test_all_three_countries_represented(self, client):
        participants = client.get_participants()
        countries_found = {p["country"] for p in participants}
        assert "Ghana" in countries_found
        assert "Sierra Leone" in countries_found
        assert "Tanzania" in countries_found


# ─── SURVEY RESPONSES ─────────────────────────────────────────────────────────

class TestSurveyResponses:
    def test_get_survey_responses_for_valid_participant(self, client):
        pid = client.get_participants()[0]["record_id"]
        responses = client.get_survey_responses(record_id=pid)
        assert len(responses) > 0

    def test_survey_response_has_required_fields(self, client):
        pid = client.get_participants()[0]["record_id"]
        responses = client.get_survey_responses(record_id=pid)
        assert len(responses) > 0
        first = responses[0]
        required = ["record_id", "month", "perceived_stress_score",
                    "anxiety_score", "depression_score", "risk_flag"]
        for field in required:
            assert field in first, f"Missing survey field: {field}"

    def test_stress_scores_are_in_range(self, client):
        pid = client.get_participants()[0]["record_id"]
        responses = client.get_survey_responses(record_id=pid)
        for r in responses:
            if "perceived_stress_score" in r:
                assert 0 <= r["perceived_stress_score"] <= 40

    def test_anxiety_scores_are_in_range(self, client):
        pid = client.get_participants()[0]["record_id"]
        responses = client.get_survey_responses(record_id=pid)
        for r in responses:
            if "anxiety_score" in r:
                assert 0 <= r["anxiety_score"] <= 21

    def test_risk_flag_is_high_or_low(self, client):
        pid = client.get_participants()[0]["record_id"]
        responses = client.get_survey_responses(record_id=pid)
        for r in responses:
            if "risk_flag" in r:
                assert r["risk_flag"] in ("HIGH", "LOW")

    def test_get_all_survey_responses_when_no_filter(self, client):
        all_responses = client.get_survey_responses()
        assert len(all_responses) > 150  # More than one per participant

    def test_filter_by_instrument(self, client):
        monthly = client.get_survey_responses(instrument="monthly_self_report")
        assert len(monthly) > 0
        assert all(r.get("redcap_repeat_instrument") == "monthly_self_report" for r in monthly)


# ─── CONSENT MANAGEMENT ───────────────────────────────────────────────────────

class TestConsent:
    def test_consent_record_exists_for_each_participant(self, client):
        participants = client.get_participants()
        for p in participants[:10]:  # spot-check first 10
            consent = client.get_consent_status(p["record_id"])
            assert consent is not None

    def test_consent_record_has_required_fields(self, client):
        pid = client.get_participants()[0]["record_id"]
        consent = client.get_consent_status(pid)
        required = ["record_id", "consent_status", "consent_date", "consent_version"]
        for field in required:
            assert field in consent, f"Missing consent field: {field}"

    def test_update_consent_changes_status(self, client):
        # Use a fresh client to avoid side-effects on shared fixture
        fresh = RedCapMockClient(seed=99)
        pid = fresh.get_participants()[0]["record_id"]
        fresh.update_consent(pid, "withdrawn")
        result = fresh.get_consent_status(pid)
        assert result["consent_status"] == "withdrawn"

    def test_update_consent_for_invalid_id_returns_error(self, client):
        fresh = RedCapMockClient(seed=99)
        result = fresh.update_consent("NEPS-INVALID-9999", "consented")
        assert "error" in result


# ─── DISTRESS SCREENINGS ──────────────────────────────────────────────────────

class TestDistressScreenings:
    def test_distress_screenings_are_list(self, client):
        screenings = client.get_distress_screenings()
        assert isinstance(screenings, list)

    def test_screening_has_required_fields(self, client):
        screenings = client.get_distress_screenings()
        if screenings:
            s = screenings[0]
            required = ["record_id", "severity", "distress_score", "resolution_status"]
            for field in required:
                assert field in s, f"Missing screening field: {field}"

    def test_severity_values_are_valid(self, client):
        valid = {RiskLevel.HIGH, RiskLevel.CRITICAL}
        for s in client.get_distress_screenings():
            assert s["severity"] in valid, f"Invalid severity: {s['severity']}"

    def test_filter_open_screenings(self, client):
        open_screenings = client.get_distress_screenings(status="open")
        assert all(s["resolution_status"] == "open" for s in open_screenings)


# ─── REFERRALS ────────────────────────────────────────────────────────────────

class TestReferrals:
    def test_create_referral_returns_referral_id(self, client):
        fresh = RedCapMockClient(seed=1)
        pid = fresh.get_participants()[0]["record_id"]
        referral = fresh.create_referral(pid, destination="District Hospital", notes="Urgent")
        assert "referral_id" in referral
        assert referral["referral_id"].startswith("REF-")

    def test_create_referral_stores_correct_fields(self, client):
        fresh = RedCapMockClient(seed=2)
        pid = fresh.get_participants()[0]["record_id"]
        referral = fresh.create_referral(pid, destination="Clinic A")
        assert referral["record_id"] == pid
        assert referral["destination"] == "Clinic A"
        assert referral["status"] == "initiated"


# ─── WP6 INTERVENTION SESSIONS ────────────────────────────────────────────────

class TestWP6Sessions:
    def test_wp6_sessions_exist_for_first_20_participants(self, client):
        participants = client.get_participants()
        for p in participants[:20]:
            sessions = client.get_wp6_sessions(p["record_id"])
            assert len(sessions) == 8, f"Expected 8 sessions for {p['record_id']}"

    def test_wp6_sessions_empty_for_non_enrolled(self, client):
        # Participants beyond index 19 are not enrolled in WP6
        participants = client.get_participants()
        if len(participants) > 20:
            non_enrolled_id = participants[20]["record_id"]
            sessions = client.get_wp6_sessions(non_enrolled_id)
            assert sessions == []

    def test_session_has_required_fields(self, client):
        participants = client.get_participants()
        sessions = client.get_wp6_sessions(participants[0]["record_id"])
        if sessions:
            s = sessions[0]
            required = ["session_number", "attendance", "engagement_level",
                        "fidelity_score", "distress_pre", "distress_post"]
            for field in required:
                assert field in s, f"Missing WP6 session field: {field}"

    def test_distress_post_is_lower_than_pre_on_average(self, client):
        """Post-session distress should be lower than pre (intervention works)."""
        participants = client.get_participants()
        all_pre = []
        all_post = []
        for p in participants[:20]:
            for s in client.get_wp6_sessions(p["record_id"]):
                all_pre.append(s["distress_pre"])
                all_post.append(s["distress_post"])
        avg_pre = sum(all_pre) / len(all_pre)
        avg_post = sum(all_post) / len(all_post)
        assert avg_post < avg_pre, "Expected post-session distress to be lower on average"


# ─── STATS & EXPORT ───────────────────────────────────────────────────────────

class TestStatsAndExport:
    def test_get_stats_returns_correct_total(self, client):
        stats = client.get_stats()
        assert stats["total_participants"] == 150

    def test_get_stats_has_all_keys(self, client):
        stats = client.get_stats()
        required_keys = ["total_participants", "by_country", "active_cohort",
                         "total_surveys", "high_risk_flags", "wp6_enrolled"]
        for key in required_keys:
            assert key in stats, f"Missing stats key: {key}"

    def test_stats_wp6_enrolled_is_20(self, client):
        stats = client.get_stats()
        assert stats["wp6_enrolled"] == 20

    def test_export_records_json_returns_list(self, client):
        records = client.export_records(format="json")
        assert isinstance(records, list)
        assert len(records) > 0

    def test_export_records_csv_returns_string(self, client):
        csv_data = client.export_records(format="csv")
        assert isinstance(csv_data, str)
        assert len(csv_data) > 0

    def test_export_records_with_field_filter(self, client):
        records = client.export_records(fields=["record_id", "month"])
        for r in records[:5]:
            assert set(r.keys()).issubset({"record_id", "month"})

    def test_export_metadata_returns_project_info(self, client):
        meta = client.export_metadata()
        assert meta["project_id"] == "NEPS-2025"
        assert "instruments" in meta
        assert "events" in meta
        assert len(meta["instruments"]) > 0
        assert len(meta["events"]) > 0
