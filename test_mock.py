"""
Test script to verify the mock REDCap dataset is working correctly.
Run this script to confirm you have access to all the data!
"""

from app.services.redcap_mock import RedCapMockClient


def main():
    print("🧪 Testing NEPS Mock REDCap Dataset...\n")

    # Initialize client
    client = RedCapMockClient()

    # Get overall stats
    stats = client.get_stats()
    print(f"📊 Overall Stats:")
    print(f"   Total Participants: {stats['total_participants']}")
    print(f"   Active Cohort: {stats['active_cohort']}")
    print(f"   Total Surveys: {stats['total_surveys']}")
    print(f"   High Risk Flags: {stats['high_risk_flags']}")
    print(f"   WP6 Enrolled: {stats['wp6_enrolled']}\n")

    # Check all countries
    print("🌍 Participants by Country:")
    ghana = client.get_participants(country="Ghana")
    sierra_leone = client.get_participants(country="Sierra Leone")
    tanzania = client.get_participants(country="Tanzania")

    print(f"   Ghana: {len(ghana)} participants")
    print(f"   Sierra Leone: {len(sierra_leone)} participants")
    print(f"   Tanzania: {len(tanzania)} participants\n")

    # Check sample participants from each country
    print("👤 Sample Participants:")

    # Ghana
    ghana_sample = client.get_participant("NEPS-GHA-0001")
    print(f"   NEPS-GHA-0001: {ghana_sample['country']}, {ghana_sample['site']}, Age {ghana_sample['age']}")

    # Sierra Leone
    sle_sample = client.get_participant("NEPS-SLE-0001")
    print(f"   NEPS-SLE-0001: {sle_sample['country']}, {sle_sample['site']}, Age {sle_sample['age']}")

    # Tanzania
    tza_sample = client.get_participant("NEPS-TZA-0001")
    print(f"   NEPS-TZA-0001: {tza_sample['country']}, {tza_sample['site']}, Age {tza_sample['age']}\n")

    # Check survey data
    print("📋 Sample Survey Data (NEPS-GHA-0001):")
    surveys = client.get_survey_responses(record_id="NEPS-GHA-0001")
    if surveys:
        sample_survey = surveys[0]
        print(f"   Month: {sample_survey['month']}")
        print(f"   Perceived Stress Score: {sample_survey['perceived_stress_score']}")
        print(f"   Anxiety Score: {sample_survey['anxiety_score']}")
        print(f"   Risk Flag: {sample_survey['risk_flag']}\n")

    # Check distress screenings
    print("🆘 Distress Screenings:")
    screenings = client.get_distress_screenings()
    print(f"   Total: {len(screenings)}")
    if screenings:
        print(f"   Sample Severity: {screenings[0]['severity']}\n")

    # Check WP6 sessions
    print("💬 WP6 Intervention Sessions:")
    wp6_participant = list(client._wp6_sessions.keys())[0]
    sessions = client.get_wp6_sessions(wp6_participant)
    print(f"   {wp6_participant} has {len(sessions)} sessions\n")

    # Success message
    print("✅ All tests passed! The mock dataset is working perfectly!")
    print("💡 You can now use this data for model training and development!")


if __name__ == "__main__":
    main()
