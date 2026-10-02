from __future__ import annotations

import pandas as pd


_EXAMPLES = {
    "Payment issue": [
        "My payment was deducted but the transaction is still pending.",
        "I sent a bank transfer yesterday and the money has not arrived.",
        "The bill payment left my account but the merchant says it is unpaid.",
        "My direct deposit is missing and I need help locating it.",
        "A card payment is pending for several days and the balance is unavailable.",
        "I was charged for a transfer that never reached the recipient.",
        "The payment failed but the funds were still taken from my account.",
        "My refund has been approved but has not been credited back yet.",
        "I made a payment twice because the first one was stuck as pending.",
        "The scheduled payment did not process even though I had enough money.",
    ],
    "Card and billing": [
        "There is an annual fee on my card statement that I do not recognize.",
        "I was charged interest even though I paid the full balance on time.",
        "My credit card statement has a duplicate charge from the same store.",
        "The minimum payment on my card changed without an explanation.",
        "I need help understanding a fee listed on my monthly statement.",
        "A late fee was added even though my payment was made before the due date.",
        "My card was declined at checkout despite having available credit.",
        "The promotional rate ended earlier than the terms said it would.",
        "I cannot see my latest credit card statement in the app.",
        "The balance on my card is higher than the purchases I made.",
    ],
    "Fraud and disputes": [
        "Someone used my account to make purchases I did not authorize.",
        "I reported identity theft but the fraudulent account is still open.",
        "My debit card was stolen and there are unauthorized transactions.",
        "The bank denied my fraud claim without investigating the charges.",
        "I need to dispute a transaction that I never made.",
        "A scammer transferred money from my account and support has not responded.",
        "My personal information was exposed and a new account appeared on my report.",
        "I do not recognize a withdrawal and need it investigated as fraud.",
        "The account was taken over and I cannot stop the unauthorized payments.",
        "I submitted documents for a fraud case but the claim was closed.",
    ],
    "Account access": [
        "I am locked out of online banking and the password reset does not work.",
        "The verification code never arrives so I cannot sign in to my account.",
        "My account was frozen and I cannot access my own money.",
        "The mobile app keeps rejecting my correct username and password.",
        "I cannot update my phone number because I cannot log in.",
        "My account was unexpectedly closed and I need to know why.",
        "The website gives an error whenever I try to access my account.",
        "I need help restoring access after changing my email address.",
        "The bank keeps sending a login code to my old phone number.",
        "I have been unable to use online banking for three days.",
    ],
    "Loans and collections": [
        "My student loan payment was applied to the wrong loan balance.",
        "The mortgage servicer says I am behind even though I paid on time.",
        "A debt collector keeps calling about an account that is not mine.",
        "My loan application has been pending for weeks with no update.",
        "I was charged a fee for paying off my auto loan early.",
        "The amount due on my mortgage increased and I did not receive notice.",
        "I requested a payment plan but the lender has not replied.",
        "My loan was transferred and the new servicer cannot find my payments.",
        "A collection account remains on my credit report after I paid it.",
        "I need help correcting an error on my loan repayment history.",
    ],
    "Customer service": [
        "I have contacted support several times and nobody has resolved my issue.",
        "The representative promised a callback but I never heard back.",
        "I was given conflicting information by two customer service agents.",
        "My complaint was closed but the problem is still happening.",
        "I cannot reach a person to discuss an issue with my account.",
        "The company has not answered my messages about the disputed charge.",
        "I submitted the requested forms but support says they cannot find them.",
        "I have waited a month for an answer to my written complaint.",
        "Customer service transferred me repeatedly without helping.",
        "The company did not explain how it handled my complaint.",
    ],
}


def make_sample_data() -> pd.DataFrame:
    """Return a small, clearly labeled corpus for an offline product preview."""
    rows = []
    for category, narratives in _EXAMPLES.items():
        for index, narrative in enumerate(narratives):
            rows.append(
                {
                    "narrative": narrative,
                    "category": category,
                    "product": category,
                    "date_received": pd.Timestamp("2025-01-01") + pd.Timedelta(index, unit="D"),
                    "company": "Illustrative financial services provider",
                }
            )
    return pd.DataFrame(rows)