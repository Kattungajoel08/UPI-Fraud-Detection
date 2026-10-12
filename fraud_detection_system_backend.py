import cv2
import time

from urllib.parse import urlparse, parse_qs, unquote

from services.risk_engine import compute_risk, save_transaction


# ============================================================
# UPI QR PARSER
# ============================================================

def parse_upi_qr(qr_data):

    try:

        parsed = urlparse(qr_data)

        if parsed.scheme != "upi":
            return None

        params = parse_qs(parsed.query)

        upi_id = params.get("pa", ["Unknown"])[0]

        name = unquote(
            params.get("pn", ["Unknown"])[0]
        )

        merchant_code = params.get(
            "mc", ["Unknown"]
        )[0]

        mode = params.get(
            "mode", ["Unknown"]
        )[0]

        purpose = params.get(
            "purpose", ["Unknown"]
        )[0]

        return {
            "upi_id": upi_id,
            "name": name,
            "merchant_code": merchant_code,
            "mode": mode,
            "purpose": purpose
        }

    except Exception:

        return None


# ============================================================
# QR SCANNER
# ============================================================

def scan_qr_code():

    cap = cv2.VideoCapture(0)

    detector = cv2.QRCodeDetector()

    print("\n📷 Opening camera...")
    print("🔍 Point the camera at a UPI QR code.")
    print("❌ Press 'q' to cancel.")

    while True:

        ret, frame = cap.read()

        if not ret:

            print("❌ Could not access camera.")

            break

        data, points, _ = detector.detectAndDecode(frame)

        # Draw QR detection box
        if points is not None:

            points = points.astype(int)

            for i in range(len(points[0])):

                cv2.line(
                    frame,
                    tuple(points[0][i]),
                    tuple(
                        points[0][
                            (i + 1) % len(points[0])
                        ]
                    ),
                    (0, 255, 0),
                    3
                )

        cv2.putText(
            frame,
            "Scan UPI QR Code",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "UPI QR Scanner",
            frame
        )

        # QR successfully detected
        if data:

            print("\n✅ QR Code Detected!")

            cap.release()

            cv2.destroyAllWindows()

            return data

        # Quit camera
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

    return None


# ============================================================
# AUTHENTICATION ENGINE
# ============================================================

def verify_pin():

    print("\n🔐 PIN VERIFICATION")

    pin = input(
        "Enter 4-digit transaction PIN: "
    )

    # DEMO ONLY
    # Never use or store a real UPI PIN.
    demo_pin = "1234"

    if pin == demo_pin:

        print(
            "✅ PIN verification successful."
        )

        return True

    print("❌ Incorrect PIN.")

    return False


def verify_otp():

    print("\n📲 OTP VERIFICATION")

    # DEMO ONLY
    demo_otp = "123456"

    print(
        "📨 Demo OTP sent to registered device."
    )

    print(
        "(For demonstration, use OTP: 123456)"
    )

    otp = input("Enter OTP: ")

    if otp == demo_otp:

        print(
            "✅ OTP verification successful."
        )

        return True

    print("❌ Invalid OTP.")

    return False


def verify_security_questions():

    print("\n🛡️ SECURITY VERIFICATION")

    print(
        "Answer the following security questions."
    )

    print("\nQuestion 1:")
    print(
        "What is your favorite programming language?"
    )

    answer1 = input(
        "Answer: "
    ).strip().lower()

    print("\nQuestion 2:")
    print(
        "What was your first project domain?"
    )

    answer2 = input(
        "Answer: "
    ).strip().lower()

    # DEMO ANSWERS ONLY
    correct_answer1 = "python"
    correct_answer2 = "fraud detection"

    if (
        answer1 == correct_answer1
        and
        answer2 == correct_answer2
    ):

        print(
            "✅ Security verification successful."
        )

        return True

    print(
        "❌ Security verification failed."
    )

    return False


# ============================================================
# TRANSACTION SIMULATION
# ============================================================

def simulate_transaction():

    print("\n")

    print("==============================================")
    print("       💳 UPI FRAUD DETECTION SYSTEM")
    print("==============================================")


    # ========================================================
    # STEP 1 — SCAN QR
    # ========================================================

    qr_data = scan_qr_code()

    if not qr_data:

        print(
            "\n❌ QR scanning cancelled."
        )

        return


    # ========================================================
    # STEP 2 — PARSE QR
    # ========================================================

    upi_info = parse_upi_qr(qr_data)

    if not upi_info:

        print(
            "\n❌ Invalid UPI QR code."
        )

        return


    print(
        "\n✅ UPI QR Code Validated"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"👤 Recipient Name   : "
        f"{upi_info['name']}"
    )

    print(
        f"📱 UPI ID           : "
        f"{upi_info['upi_id']}"
    )

    print(
        f"🏪 Merchant Code    : "
        f"{upi_info['merchant_code']}"
    )

    print(
        f"💳 Payment Mode     : "
        f"{upi_info['mode']}"
    )

    print(
        "----------------------------------------------"
    )


    # ========================================================
    # STEP 3 — ENTER AMOUNT
    # ========================================================

    while True:

        try:

            amount = float(
                input(
                    "\n💰 Enter Transaction Amount (₹): "
                )
            )

            if amount <= 0:

                print(
                    "❌ Amount must be greater than zero."
                )

                continue

            break

        except ValueError:

            print(
                "❌ Invalid amount. "
                "Please enter a number."
            )


    # ========================================================
    # STEP 4 — FRAUD ANALYSIS
    # ========================================================

    print(
        "\n🔄 Processing Transaction..."
    )

    time.sleep(1)

    upi_id = upi_info["upi_id"]

    result = compute_risk(
        amount,
        upi_id
    )

    score = result["risk_score"]

    risk = result["risk"]

    reason = result.get(
        "reason",
        "Risk analysis completed"
    )


    # ========================================================
    # STEP 5 — FRAUD ANALYSIS RESULT
    # ========================================================

    print("\n")

    print(
        "=============================================="
    )

    print(
        "          📊 FRAUD ANALYSIS"
    )

    print(
        "=============================================="
    )

    print(
        f"💰 Transaction Amount : ₹{amount:.2f}"
    )

    print(
        f"👤 Recipient          : "
        f"{upi_info['name']}"
    )

    print(
        f"📱 UPI ID             : "
        f"{upi_id}"
    )

    print(
        f"🧠 Risk Score         : "
        f"{score:.3f}"
    )

    print(
        f"🚨 Risk Level         : "
        f"{risk}"
    )

    print(
        f"💡 Risk Reason        : "
        f"{reason}"
    )


    # ========================================================
    # STEP 6 — DECISION ENGINE
    # ========================================================

    print(
        "\n⚙️ Decision Engine"
    )

    print(
        "----------------------------------------------"
    )


    if risk == "LOW":

        decision = "APPROVED"

        print(
            "🟢 LOW RISK TRANSACTION"
        )

        print(
            "🔐 Required Authentication: PIN"
        )


    elif risk == "MEDIUM":

        decision = "VERIFICATION_REQUIRED"

        print(
            "🟡 MEDIUM RISK TRANSACTION"
        )

        print(
            "🔐 Required Authentication: PIN + OTP"
        )


    else:

        decision = "BLOCKED"

        print(
            "🔴 HIGH RISK TRANSACTION"
        )

        print(
            "🔐 Required Authentication: "
            "PIN + OTP + Security Questions"
        )


    # ========================================================
    # STEP 7 — AUTHENTICATION
    # ========================================================

    print(
        "\n🔐 AUTHENTICATION ENGINE"
    )

    print(
        "----------------------------------------------"
    )

    authentication_success = False


    # --------------------------------------------------------
    # LOW RISK → PIN
    # --------------------------------------------------------

    if risk == "LOW":

        print(
            "🟢 LOW RISK"
        )

        print(
            "🔑 Authentication Level: PIN"
        )

        authentication_success = verify_pin()


    # --------------------------------------------------------
    # MEDIUM RISK → PIN + OTP
    # --------------------------------------------------------

    elif risk == "MEDIUM":

        print(
            "🟡 MEDIUM RISK"
        )

        print(
            "🔑 Authentication Level: PIN + OTP"
        )

        pin_verified = verify_pin()

        if pin_verified:

            otp_verified = verify_otp()

            authentication_success = (
                otp_verified
            )


    # --------------------------------------------------------
    # HIGH RISK → PIN + OTP + SECURITY
    # --------------------------------------------------------

    else:

        print(
            "🔴 HIGH RISK"
        )

        print(
            "🔑 Authentication Level:"
        )

        print(
            "   PIN + OTP + Security Questions"
        )

        pin_verified = verify_pin()

        if pin_verified:

            otp_verified = verify_otp()

            if otp_verified:

                security_verified = (
                    verify_security_questions()
                )

                authentication_success = (
                    security_verified
                )


    # ========================================================
    # STEP 8 — FINAL TRANSACTION STATUS
    # ========================================================

    print(
        "\n💳 FINAL TRANSACTION STATUS"
    )

    print(
        "----------------------------------------------"
    )


    if authentication_success:

        if risk == "LOW":

            final_status = "APPROVED"

        elif risk == "MEDIUM":

            final_status = (
                "APPROVED_AFTER_VERIFICATION"
            )

        else:

            final_status = (
                "APPROVED_AFTER_ENHANCED_VERIFICATION"
            )

        print(
            "✅ Authentication Successful"
        )

        print(
            f"💳 Transaction {final_status}"
        )


    else:

        final_status = "DECLINED"

        print(
            "❌ Authentication Failed"
        )

        print(
            "🛑 Transaction Declined"
        )


    # ========================================================
    # STEP 9 — SYSTEM INSIGHT
    # ========================================================

    print(
        "\n🧠 System Insight"
    )

    print(
        "----------------------------------------------"
    )


    if amount >= 5000:

        print(
            "➡ High transaction amount "
            "increased fraud probability."
        )

    elif amount >= 2000:

        print(
            "➡ Moderate transaction amount "
            "contributed to risk."
        )

    else:

        print(
            "➡ Low transaction value detected."
        )


    print(
        f"➡ {reason}"
    )


    # ========================================================
    # STEP 10 — SAVE TRANSACTION
    # ========================================================

    save_transaction(
        amount,
        upi_id,
        score,
        risk,
        final_status
    )

    print(
        "\n💾 Transaction stored in database."
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print(
        "=============================================="
    )

    print(
        f"             DECISION: {final_status}"
    )

    print(
        "=============================================="
    )

    print(
        "\n👋 Transaction analysis completed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    simulate_transaction()