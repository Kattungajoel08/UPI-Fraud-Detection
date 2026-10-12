import time
import random

# ---------------- SIMPLE RISK FUNCTION ----------------
def compute_risk(amount):
    if amount < 2000:
        return 0.2, "LOW"
    elif amount < 5000:
        return 0.5, "MEDIUM"
    else:
        return 0.8, "HIGH"

# ---------------- OTP SYSTEM ----------------
def otp_verification():
    otp = random.randint(100000, 999999)
    print(f"\n📩 OTP Sent: {otp}")

    user = input("Enter OTP: ")
    if user == str(otp):
        print("✅ OTP Verified")
        return True
    else:
        print("❌ Wrong OTP")
        return False

# ---------------- PIN ----------------
def verify_pin():
    pin = input("Enter PIN: ")
    return pin == "1234"

# ---------------- MAIN SIMULATION ----------------
def simulate_transaction():
    print("\n💳 UPI Fraud Detection (Simulation)")
    print("----------------------------------")

    while True:
        try:
            amount = float(input("\nEnter Transaction Amount (₹): "))
        except ValueError:
            print("❌ Enter a valid number")
            continue

        print("\n🔄 Processing...")
        time.sleep(1)

        score, risk = compute_risk(amount)

        print("\n📊 --- ANALYSIS ---")
        print(f"💰 Amount       : ₹{amount}")
        print(f"🧠 Risk Score   : {score}")
        print(f"🚨 Risk Level   : {risk}")

        print("\n⚙️ Decision Engine:")

        if risk == "LOW":
            print("✅ Transaction Approved")

        elif risk == "MEDIUM":
            print("⚠ OTP + PIN Required")
            if otp_verification() and verify_pin():
                print("✅ Transaction Approved")
            else:
                print("❌ Transaction Blocked")

        else:
            print("🚨 High Risk Transaction")
            if otp_verification() and verify_pin():
                print("⚠ Extra Verification Needed")
                answer = input("Security Question (Pet name?): ")
                if answer.lower() == "tommy":
                    print("✅ Transaction Approved")
                else:
                    print("❌ Transaction Blocked")
            else:
                print("❌ Transaction Blocked")

        # Continue or exit
        choice = input("\nDo another transaction? (y/n): ").lower()
        if choice != 'y':
            print("\n👋 Exiting Simulation")
            break


if __name__ == "__main__":
    simulate_transaction()