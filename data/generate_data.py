"""
Generates a synthetic IT support ticket dataset for the auto-resolution
prototype. In production this would be replaced by real historical
ticket exports (2M tickets/month, 5,000 issue types). Here we simulate
a smaller but structurally identical dataset:

    coarse_category -> fine_issue_type -> (ticket_text, resolution)

30% of tickets are perturbed with noise (typos, missing punctuation,
truncation) to simulate poorly-written tickets, matching the stated
constraint.
"""
import random
import csv
import re

random.seed(42)

# coarse category -> list of (issue_type, [template ticket phrasings], resolution)
CATALOG = {
    "Network": [
        ("vpn_connection_failure",
         ["VPN keeps disconnecting every few minutes",
          "Cannot connect to company VPN, error code 809",
          "VPN client says authentication failed but password is correct",
          "unable to establish vpn tunnel from home network"],
         "Restart the VPN client, confirm correct server region, and re-authenticate via SSO. "
         "If error 809 persists, check that UDP port 500/4500 is not blocked by local firewall."),
        ("wifi_no_internet",
         ["connected to office wifi but no internet access",
          "Wifi shows connected with no internet, exclamation mark on icon",
          "laptop wont get an ip address on the corporate wifi",
          "internet dropping every 10 mins on wireless"],
         "Forget and rejoin the WiFi network, renew DHCP lease (ipconfig /release && /renew), "
         "and confirm the access point is on the latest firmware."),
        ("slow_network_speed",
         ["internet is extremely slow today",
          "file transfers to shared drive are taking forever",
          "network speed test shows less than 1mbps",
          "video calls keep freezing due to slow connection"],
         "Run a speed test from a wired connection to isolate WiFi vs WAN issues, "
         "check for bandwidth-heavy background syncs, and escalate to network team if WAN saturation confirmed."),
    ],
    "Access_Login": [
        ("password_reset",
         ["forgot my password and cant login",
          "account locked after too many login attempts",
          "need password reset for email account",
          "locked out of my workstation, wrong password too many times"],
         "Verify identity via registered MFA device, then trigger self-service password reset portal. "
         "If account is locked, unlock via IAM console after identity verification."),
        ("mfa_issues",
         ["not receiving mfa code on my phone",
          "authenticator app out of sync, codes rejected",
          "lost my phone and cant complete two factor authentication",
          "mfa push notification never arrives"],
         "Re-sync the authenticator app time, or issue a temporary bypass code after identity verification. "
         "For lost devices, de-register old device and enroll a new one via the security portal."),
        ("sso_login_error",
         ["single sign on redirect loop when logging into portal",
          "sso says invalid session please try again",
          "cant access okta dashboard, blank page after login",
          "sso token expired immediately after login"],
         "Clear browser cookies/cache for the identity provider domain, confirm system clock is correct, "
         "and retry. If persistent, check IdP status page for outages."),
    ],
    "Hardware": [
        ("laptop_wont_boot",
         ["laptop wont turn on at all",
          "black screen on startup, no bios logo",
          "computer stuck on boot loop restarting itself",
          "pressed power button nothing happens"],
         "Confirm power adapter and battery health, attempt a hard reset (hold power 15s), "
         "and if no POST, escalate for hardware diagnostic/replacement."),
        ("printer_not_working",
         ["printer not responding from my computer",
          "print job stuck in queue forever",
          "printer showing offline even though its on",
          "prints come out blank"],
         "Clear the print queue and restart the print spooler service, reinstall printer driver if offline persists, "
         "check toner/ink levels for blank output."),
        ("monitor_no_display",
         ["external monitor not detected when docked",
          "second monitor shows no signal",
          "screen flickering constantly on external display",
          "monitor resolution keeps resetting"],
         "Reseat the display cable, verify dock firmware is current, and force-detect displays via "
         "display settings; swap cable/port to isolate faulty hardware."),
    ],
    "Software": [
        ("app_crash",
         ["outlook keeps crashing on startup",
          "excel closes unexpectedly when opening large files",
          "application freezes and has to be force closed",
          "software crashes with an unhandled exception error"],
         "Clear the application cache/profile, repair the Office installation, "
         "and check Event Viewer for the specific exception to identify a faulty add-in."),
        ("software_install_request",
         ["need access to install adobe photoshop",
          "requesting installation of visual studio code",
          "cant install software, admin rights required",
          "need a license for project management tool"],
         "Submit software request through the self-service catalog for auto-approved titles; "
         "for licensed software, route to procurement for license allocation before install."),
        ("update_failure",
         ["windows update stuck at 45 percent for hours",
          "update failed with error 0x80070002",
          "cannot install latest security patch",
          "software update keeps failing and rolling back"],
         "Run the Windows Update troubleshooter, clear the SoftwareDistribution cache, "
         "and retry; for persistent 0x8007xxxx errors, run DISM /RestoreHealth."),
    ],
    "Account_Access_Data": [
        ("shared_drive_access",
         ["dont have access to the finance shared drive",
          "permission denied when opening team folder",
          "need access added to project shared drive",
          "cant see the shared drive that my team uses"],
         "Verify correct security group membership in AD/IAM, request manager approval if not yet a member, "
         "and add user to the relevant access group."),
        ("email_not_syncing",
         ["outlook not syncing new emails",
          "emails delayed by several hours",
          "sent items not showing up in outlook",
          "mailbox stuck reconnecting to server"],
         "Check mailbox size against quota, repair the Outlook profile/OST file, "
         "and verify Exchange server connectivity status."),
        ("data_recovery_request",
         ["accidentally deleted an important file need it back",
          "need to restore a folder from last week",
          "lost files after a sync error",
          "need backup restore for deleted documents"],
         "Check the recycle bin/OneDrive version history first; if unavailable, "
         "submit a backup restore request with file path and approximate deletion date."),
    ],
}

NOISE_TYPOS = {
    "the": "teh", "and": "adn", "internet": "internert", "password": "passwrod",
    "cannot": "cant", "keeps": "keps", "connection": "conection", "access": "acess",
    "computer": "compter", "printer": "prnter", "please": "pls", "issue": "isue",
}


def inject_noise(text: str) -> str:
    """Simulate a poorly-written ticket: typos, dropped punctuation, casing issues, truncation."""
    words = text.split()
    out = []
    for w in words:
        lw = w.lower()
        if lw in NOISE_TYPOS and random.random() < 0.5:
            out.append(NOISE_TYPOS[lw])
        else:
            out.append(w)
    text = " ".join(out)
    if random.random() < 0.4:
        text = text.lower()
    text = re.sub(r"[.,!?]", "", text) if random.random() < 0.5 else text
    if random.random() < 0.3:
        cut = max(5, int(len(text.split()) * 0.7))
        text = " ".join(text.split()[:cut])
    if random.random() < 0.2:
        text += " asap plz help ticket urgent"
    return text


def generate(n_rows: int = 1200, noisy_fraction: float = 0.3):
    rows = []
    ticket_id = 1000
    categories = list(CATALOG.items())
    while len(rows) < n_rows:
        coarse, issues = random.choice(categories)
        issue_type, phrasings, resolution = random.choice(issues)
        base_text = random.choice(phrasings)
        is_noisy = random.random() < noisy_fraction
        text = inject_noise(base_text) if is_noisy else base_text
        rows.append({
            "ticket_id": ticket_id,
            "text": text,
            "coarse_category": coarse,
            "issue_type": issue_type,
            "resolution": resolution,
            "is_noisy": is_noisy,
        })
        ticket_id += 1
    return rows


if __name__ == "__main__":
    rows = generate()
    with open("data/tickets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic tickets to data/tickets.csv")
