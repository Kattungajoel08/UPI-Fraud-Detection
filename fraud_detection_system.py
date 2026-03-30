import time
from services.risk_engine import compute_risk

def simulate_transaction():
    print("\n💳 UPI Fraud Detection Backend Simulation")
    print("----------------------------------------")

    while True:
        try:
            amount = float(input("\nEnter Transaction Amount (₹): "))
        except:
            print("❌ Invalid input. Please enter a number.")
            continue

        print("\n🔄 Processing Transaction...")
        time.sleep(1)

        result = compute_risk(amount)
        score = result["risk_score"]
        risk = result["risk"]

        print("\n📊 --- BACKEND ANALYSIS ---")

        print(f"💰 Amount Entered     : ₹{amount}")
        time.sleep(0.5)

        print(f"🧠 Risk Score         : {round(score, 3)}")
        time.sleep(0.5)

        print(f"🚨 Risk Level         : {risk}")
        time.sleep(0.5)

        # Decision logic (same as your app behavior)
        print("\n⚙️ Decision Engine:")

        if risk == "LOW":
            print("✅ Transaction Approved (No Verification)")
        elif risk == "MEDIUM":
            print("⚠ OTP + PIN Verification Required")
        else:
            print("🚨 High Risk! Transaction Blocked / Extra Verification")

        # Optional explanation (very impressive)
        print("\n🧠 System Insight:")
        if amount > 5000:
            print("➡ High amount increases fraud probability")
        elif amount > 2000:
            print("➡ Moderate amount risk detected")
        else:
            print("➡ Low transaction value")

        # Continue loop
        choice = input("\nDo another transaction? (y/n): ")
        if choice.lower() != "y":
            print("\n👋 Exiting Simulation")
            break


if __name__ == "__main__":
    simulate_transaction()