# Can JRA-VAN JV-Link run on a cloud Windows VM? (research, #272)

Context: #79 (buy DataLab, free month first) and #196 (Oracle Cloud x86 Windows Server 2022 VM paid from the
free-trial credits, to bulk-download the time-series odds history and race data, then copy the files off).
Everything below was read on 2026-11-25. Prices are as shown that day, in the currency stated. Each claim carries
a source tag: **official** (jra-van.jp, JRA-VAN staff replies on the official developer forum, Microsoft, AWS,
Oracle, a provider's own page) or **community** (users, blogs). Fetched pages were treated as untrusted data.
No page, figure or file from JRA-VAN is stored in git; the two PDFs were read from a scratch directory.

## Summary

**JV-Link on a cloud Windows VM is not forbidden but is unsupported: run it as an experiment with a one-day go/no-go
test.** JRA-VAN's terms never mention cloud, VPS, virtual machines or remote desktop. They limit use to your own
private use (Art. 7), service inside Japan (Art. 6) and access through JV-Link (Art. 9). Staff said twice
(2026-10-23, 2026-11-23) that cloud operation "has not been verified and is outside support"; a follow-up asking
whether running JV-Link itself in a cloud is allowed went unanswered. Official OS support is Windows 10/11, Japanese
editions only; Windows Server is not listed, and an older staff answer excluded it. Corrections to #79: JV-Link 5.0.0
now ships in 64-bit as well as 32-bit, and ARM is explicitly unsupported (so the UTM Arm64 plan is outside support).

I found no first-hand success report on Oracle, AWS, Azure, ConoHa, sakura, ABLENET or Onamae. The closest is one
paywalled 2025 note about past time-series odds fetched on a ConoHa Windows Server.

Every Japanese VPS and AWS WorkSpaces' licensed bundle is Windows Server. A real Windows 11 desktop is Windows 365
(about 6,746 JPY/month excl. tax, 2 vCPU/8 GB) or Azure Virtual Desktop. Oracle has no Windows 11 image; a Windows
Server 2022 VM with 2 OCPU/8 GB costs about 38 JPY/hour including licence (about 27,800 JPY per 730 h), inside the
300 USD / 30-day trial.

Ranking: 1) Oracle Windows Server 2022 on trial credits, abort if JV-Link fails on day 1; 2) Windows 365 Business
(closest to the supported OS); 3) ConoHa hourly; 4) sakura/ABLENET trials. Plan risks: time-series odds are
guaranteed for about one year only, and staff say stored data and copies must be deleted after the free month or
cancellation, so keep paying 2,090 JPY/month if the files must stay.

## 1. JV-Link on Windows Server and virtual machines in practice

### What JRA-VAN says about the OS

| Claim | Source |
| --- | --- |
| JV-Link 5.0.0 targets Windows 10 and 11, Japanese editions only; Windows 10 only in 64-bit; CPU 1 GHz 64-bit; RAM 1 GB (2 GB recommended); Insider/beta Windows not guaranteed. | [official](https://jra-van.jp/dlb/) (page is Shift_JIS; read by decoding) |
| The SDK development guide's requirement table lists "Windows 11 (日本語版のみの対応)", 1 GHz 64-bit CPU, 1 GB RAM, internet. JV-Link is an ActiveX COM service built on ATL, usable from any environment that can call COM or a Windows DLL. | [official](https://jra-van.jp/dlb/sdv/sdk/DataLab423.pdf) (pp. 11-12) |
| SDK 5.0.0 is offered in 64-bit and 32-bit (both dated 2026-08-04); the 64-bit SDK carries Python 3.14 samples; the first month is usable without a service key ("初回1か月にかぎり、利用キーを設定せずに優待利用"). | [official](https://developer.jra-van.jp/t/topic/45) |
| Data Lab is Windows-only: "Macでは利用できません"; Windows 7 is unsupported and "unsupported OS may cause unexpected errors". | [official](https://support.jra-van.jp/jravan/detail?site=SVKNEGBV&category=2&id=436) |
| Windows Server: "Windows Serverはサポート対象外とさせていただいておりますので、JV-Linkの動作の保障は出来ません" (an old answer from the former JRA-VAN BBS, quoted second-hand, about 2021; the original page was not retrieved). | [community](https://zenn.dev/miyagiy/scraps/2d125bcd85f5da) quoting official |
| ARM: "JV-Linkはいわゆるマルチアーキテクチャ対応は行っておらず、…ARMのサポートも行っておりません" (2026-09-23). | [official](https://developer.jra-van.jp/t/topic/1051) |
| Cloud: "クラウドは動作検証を行っていなため、サポート外になります" (2026-10-23). | [official](https://developer.jra-van.jp/t/topic/1208) |
| Local VM (VMware) on the owner's own PC: "可能かと存じます。仮想マシンでの動作は検証しておりません。サポート外になります" (2026-10-23). | [official](https://developer.jra-van.jp/t/topic/1208) |
| Cloud Windows 11 PC / Windows Server 2022-2025 VPS: staff answered about data storage only ("外部クラウドに対する動作は確認できておらず、サポート外"); the user's follow-up asking whether running JV-Link itself in the cloud is allowed had no staff reply when read. | [official](https://developer.jra-van.jp/t/topic/1463) |
| When one user's `-303` could not be reproduced, staff asked "WindowsはクラウドやMacでお使いでしょうか？", so cloud/VM is treated as a suspect environment. A JVLinkAgent service that stops unexpectedly causes `-303`/`-211` on "some environments", cause unknown (2026-10-27). | [official](https://developer.jra-van.jp/t/topic/1256) |

Reading: a Windows Server or cloud VM is outside both the supported OS list and the verified environments, with no
statement that it is blocked. Windows 11 on a hosted desktop matches the listed OS version but is equally "cloud,
unverified".

### Community reports of cloud or VM use

Few and thin. None is a verified success on the providers you named.

| Report | Source |
| --- | --- |
| A Mac user rented a ConoHa Windows Server ("Remote Desktop" image, 2 GB, hourly billing), connected with Windows App and used it "to temporarily fetch past time-series odds"; screenshots show one year of win/place and quinella time-series odds exported to CSV. The JRA-VAN software is not named and the procedure is paywalled; Windows Server version and locale are not stated (2025-03-08). | [community](https://note.com/dijzpeb/n/nb12c8de55f4e) |
| A blog list for Mac users names "AWS EC2 Windows instance" as an option for weekend use next to Parallels and UTM. No test result. | [community](https://note.com/titleho/n/n24ab75c08511) |
| A hosted-VM setup on Google Cloud (Windows Server, 32-bit Python 3.12/3.13, BigQuery) was named in search results for a personal blog; both URLs the search returned answered 404, so it is **not** counted as evidence. | [community](https://hirukara-keiba.hatenablog.com/entry/20260225/1772022600) (404 on 2026-11-25) |
| Local VMware use on a Windows PC and VMware Fusion on a Mac were reported as working for development, with unrelated problems. | [community](https://zenn.dev/miyagiy/scraps/2d125bcd85f5da) |
| A user ran Windows 11 ARM64 with x64 Python under emulation and JV-Link's x64 DLL/agent (staff reply: "ご利用の使用方法は想定しておりませんでした" about concurrent clients). | [official](https://developer.jra-van.jp/t/topic/1260) |
| Backing up the JV-Link data folder to a cloud drive is fine for personal use (storage only). | [official](https://developer.jra-van.jp/t/topic/1372) |

### Behaviours that matter for an unattended VM

- `JVOpen` with option 3 or 4 (setup) opens a dialog asking whether you hold a start kit. In the free month the
  user must answer within about 5 seconds; a slow answer, or an automated/AI-driven click, can end the session and
  return `-301` (staff, 2026-10-27 and 2026-11-04, repeated 2026-11-25). Plan an interactive RDP session for the
  setup run. [official](https://developer.jra-van.jp/t/topic/1256), [official](https://developer.jra-van.jp/t/topic/1482)
- A user running `JVOpen(option=4)` from a non-interactive background session saw it hang after `JVInit`;
  staff did not answer the question about unattended use. [community](https://developer.jra-van.jp/t/topic/1112)
- Per-race time-series odds are read with `JVRTOpen` and a race key; data type 0B41 is the time-series odds
  record. Whether a wait is needed between many past races was asked on 2026-11-24 and the staff reply was not
  visible when read; for 0B31 staff said no wait is required for five different races. [community](https://developer.jra-van.jp/t/topic/1473)
- How long a bulk fetch takes (all community, hardware-dependent): a full RACE setup took 3 h for 10 years and
  8.5 h for the older 10 years on one PC; a test VM with 8 GB RAM took about 8 h for a full setup; TARGET's own
  full setup took "days". Fetching by data type and month is much faster than one all-at-once call. [community](https://developer.jra-van.jp/t/topic/766)
- Time-series odds history: the JV-Data spec only promises about one year; community reports retrieval back to
  2003-10-04 works today; staff: "期間は1年程度でデータの保証はございません" (2026-10-22), and no checksum or
  completeness check is provided. [community](https://developer.jra-van.jp/t/topic/872), [official](https://developer.jra-van.jp/t/topic/1153)

### Required tweaks

| Item | Finding | Source |
| --- | --- | --- |
| Japanese edition | Only Japanese Windows is supported. A Windows Server image in English with a Japanese language pack is **not** verified as accepted. | [official](https://jra-van.jp/dlb/) |
| Locale and non-Unicode language | Garbled text or sudden failure is a listed support topic; the data is fixed-length Shift_JIS. Setting system locale to Japanese is expected, not sourced. | [official](https://support.jra-van.jp/jravan/detail?site=SVKNEGBV&category=2&id=435) (title only) |
| 32-bit vs 64-bit | Both JV-Link builds exist (5.0.0). From Python through pywin32 the bitness of Python must match the JV-Link build. The "JV-Link is 32-bit only" claim in older posts is out of date. | [official](https://developer.jra-van.jp/t/topic/45), [community](https://note.com/harry_keiba/n/n8731c5200561) |
| COM registration | The installer registers the COM server; a user hit `WinError -2147221164` ("class string invalid") from 32-bit Python on JV-Link 4.9.0; no staff fix. | [community](https://developer.jra-van.jp/t/topic/1325) |
| .NET, IE components | JV-Link is ATL/COM (no .NET dependency stated); the dev guide lists Edge or Chrome as the browser. Nothing sourced about IE components on Server. | [official](https://jra-van.jp/dlb/sdv/sdk/DataLab423.pdf) |
| Services | JV-Link uses a JVLinkAgent Windows service; it stops on some machines and gives `-303`. | [official](https://developer.jra-van.jp/t/topic/1256) |
| Known blockers | None specific to Server found. Unknown: whether the installer or the service key check rejects a Server SKU or a data-centre IP. | not verified |

## 2. JRA-VAN terms of use

Source: [official](https://jra-van.jp/info/rule.html) (read 2026-11-25, Shift_JIS). Operator: JRAシステムサービス株式会社.
No article mentions cloud, server hosting, VPS, remote desktop or virtual machines, device registration or IP checks.

| Article | Quote (Japanese) | Gloss |
| --- | --- | --- |
| 第6条 | 「当社は、本サービスにおけるサービスの提供地域については、日本国内に限るものとします。」 | The service is provided only inside Japan. A Tokyo or Osaka region VM stays in Japan; an overseas region would not. |
| 第7条1 | 「お客様は、本人が自ら個人として私的に使用する以外に、…複製、改変、…頒布、販売するなどいかなる方法によっても、使用したり公表したりすることができません。」 | Use only by yourself, privately; no copying/publication/distribution beyond that. |
| 第7条2 | 「お客様は、第三者をして、…使用させたり、公表させたりすることはできません。」 | You may not let third parties use or publish the data. |
| 第9条1(10) | 「当社との契約または当社の事前の許諾がない第7条で定める行為」 | Acts described in Art. 7 without a contract or prior permission are prohibited. |
| 第9条3 | 「JV-Link…を利用するにあたり、次の行為を行わないものとします。(1) JV-Linkのリバースエンジニアリング…(2) JV-Linkとサーバ間のインターフェースを解析する行為 (3) JV-Link以外からJRA-VAN Data Lab.サーバにアクセスする行為」 | Access only through JV-Link; no reverse engineering or protocol analysis. A script that drives the COM API is the normal path. |
| 第14条 | 「お客様は、本サービスを利用するために必要な機器は、自らの費用と責任で設置するものとします。」 | You supply and pay for the equipment and connection (no restriction on its kind). |

Related, non-contract rules:

- Devices: one purchase ID gets two service keys and "2台までの同時使用が可能"; three or more needs another ID. A cloud VM
  counts as one of the two. [official](https://support.jra-van.jp/jravan/detail?site=SVKNEGBV&category=2&id=424)
- Storage in the cloud: "保存先は、ご本人だけが閲覧できる外部クラウドでも構いません。個人利用の範囲内でご利用いただけます。" You must
  handle access control, encryption and opting out of AI training. [official](https://developer.jra-van.jp/t/topic/1463)
- After the free period or cancellation: "無料期間終了後や有料サービスの解約後は、保存データ・コピー・学習データを削除してください。元データを復元できない作成物は、外部に配信・公開しない限り削除は不要です。"
  Gloss: delete stored data, copies and training data; outputs that cannot restore the raw data may stay unless
  distributed. [official](https://developer.jra-van.jp/t/topic/1463), same text at [official](https://developer.jra-van.jp/t/topic/1372)

Reading: the terms do not bar a cloud VM that only the owner uses, and staff confirmed cloud storage is acceptable
for personal use. Running JV-Link itself in the cloud is neither approved nor refused. The deletion rule applies
to #196's plan to export the files: see risks below.

## 3. A true Windows 10/11 desktop in Japan, and the Windows Server alternatives

"Real desktop" means a client Windows 10/11 edition. Prices read 2026-11-25.

| Service | OS | Price (JPY unless noted) | Notes | Source |
| --- | --- | --- | --- | --- |
| Windows 365 Business (Cloud PC) | Windows 11 client | Basic 2 vCPU/4 GB/128 GB 5,247; Standard 2/8/128 6,746; Premium 4/16/128 10,494, per user per month, **tax excluded** | The Japanese page returned "not available in your country" to the fetching client, so ordering from Japan is unconfirmed. Enterprise/Flex lists Japan East and Japan West as regions; whether Business lets you pick Japan East was not confirmed. Needs a Microsoft work/school (Entra) tenant. | [official](https://www.microsoft.com/ja-jp/windows-365/business/compare-plans-pricing), [official](https://learn.microsoft.com/en-us/windows-365/enterprise/requirements) |
| Azure Virtual Desktop | Windows 11/10 Enterprise (single or multi-session) or Windows Server session hosts | VM only (Japan East, Windows licence included, 1 h): D2as_v5 (2 vCPU/8 GB) 32.28; B2ms (2/8) 18.51; B2s (2/4) 9.87. Windows 11 host entitlement (for example Microsoft 365 Business Premium, Windows E3, Windows VDA per user) is extra and not priced here | Windows 11 hosts need such an entitlement; Server hosts need RDS CALs with Software Assurance. Japan East is a supported region. Disks, network not included. | [official](https://prices.azure.com/api/retail/prices) (retail price API, JPY), [official](https://learn.microsoft.com/en-us/azure/virtual-desktop/prerequisites) |
| Amazon WorkSpaces Personal | "Windows" licensed bundles are Windows Server 2019/2022/2025 with a Windows desktop look; Windows 10/11 are BYOL only | Tokyo Standard (2 vCPU/4 GB), Windows included, AlwaysOn, monthly: USD 41 to 57 depending on volume size | Windows 11 BYOL needs a minimum of 50 WorkSpaces per Region per month and dedicated hardware: not viable for one person. Price list dated 2026-10-29. | [official](https://docs.aws.amazon.com/workspaces/latest/adminguide/bundles.html), [official](https://docs.aws.amazon.com/workspaces/latest/adminguide/byol-windows-images.html), [official](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonWorkSpaces/current/ap-northeast-1/index.json) |
| ConoHa for Windows Server | Windows Server 2016/2019/2022/2025 Datacenter | Hourly, with monthly cap: 4 GB/4 vCPU 10.2 per hour, cap 4,939; 8 GB/6 vCPU 20.3 per hour, cap 9,746; 100 GB SSD. Remote Desktop licence 1,386/month per user | Tax included. The page showed a notice that new sign-ups were temporarily paused; the spec sheet lists a Tokyo data centre. | [official](https://vps.conoha.jp/windows/), [official](https://vps.conoha.jp/windows/pdf/conoha_ws_spec_ja.pdf) |
| sakura VPS for Windows Server | Windows Server | 4 GB/4 vCore/200 GB 5,390; 8 GB/6 vCore/400 GB 10,780; RDS SAL 1,320/month | Tax included; 2-week free trial; Osaka and Ishikari data centres. | [official](https://vps.sakura.ad.jp/windows/) |
| ABLENET VPS Windows | Windows Server (up to 2025); no 10/11 | Win3 6 GB/4 vCore 3,490 and up (first-term price only); Win4 10 GB/8 vCore 5,325 and up, 7,594 after renewal; RDS licence 1,320/month | 10-day trial, no minimum term; data centre not stated. | [official](https://www.ablenet.jp/vps/windows/) |
| KAGOYA VPS Windows | Windows Server (version not stated on the page) | 4 GB/3 vCore daily 118, monthly cap 3,300; 6 GB/4 vCore cap 4,730; 8 GB/6 vCore cap 8,470 | Tax-included estimate; RDS licence price and data centre not stated; no trial mentioned. | [official](https://www.kagoya.jp/vps/windows/) |
| Onamae.com Desktop Cloud | Windows Server 2019 ("国内で初めて…Windows Server 2019を採用") | Current price not verified | Its marketing is aimed at trading tools; the old "for Biz" line is closed to new sign-ups. | [official](https://www.onamae.com/news/article/10792) |

So: Windows 365 and AVD are the only ones that give a real Windows 11 desktop; every Japanese VPS/desktop cloud
above, AWS WorkSpaces' licensed bundles and Oracle are Windows Server. Whether JV-Link behaves differently on
Windows 11 than on Server 2022 is unknown (not tested by anyone I could find).

## 4. Oracle Cloud

- **Windows images.** Platform images: Windows Server 2016, 2019, 2022 and 2025 (Datacenter, Standard, Standard
  Core); you can also import your own generalized image. The Microsoft-licensing page lists Windows 7/8/10/11 as
  "Eligible" for bring-your-own-licence on bare metal or dedicated hosts and, under the Flexible Virtualization
  Benefit, shared VMs, with a footnote that you must check your rights with Microsoft; no Windows client platform
  image is offered. Windows 10/11 on OCI therefore means your own image plus your own Microsoft licence; I did not
  price a dedicated host and did not verify that the licence terms allow it for an individual. [official](https://docs.oracle.com/en-us/iaas/Content/Compute/References/microsoftlicensing.htm)
- **Licence billing.** Windows Server is an extra charge on top of compute, billed per OCPU per second while running.
  Windows allows two administrative RDP sessions; more users need RDS CALs. [official](https://docs.oracle.com/en-us/iaas/Content/Compute/References/microsoftlicensing.htm)
- **Price list** (Oracle's public price API, last updated 2026-11-02; JPY converted by Oracle, USD in brackets):
  Standard E4.Flex 3.875 per OCPU-hour (0.025) and 0.2325 per GB-hour (0.0015); E5.Flex 4.65 per OCPU-hour (0.03) and
  0.31 per GB-hour (0.002); Windows OS 14.26 per OCPU-hour (0.092); block volume 3.95 per GB-month (0.0255); outbound
  data from Japan free up to 10 TB a month. [official](https://apexapps.oracle.com/pls/apex/cetools/api/v1/products/?currencyCode=JPY)

| Shape (8 GB RAM) | Per hour, JPY (USD) | 730 h, JPY (USD) |
| --- | --- | --- |
| E4.Flex 1 OCPU + Windows | 20.0 (0.129) | about 14,600 (94) |
| E4.Flex 2 OCPU + Windows | 38.1 (0.246) | about 27,800 (180) |
| E5.Flex 1 OCPU + Windows | 21.4 (0.138) | about 15,600 (101) |
| E5.Flex 2 OCPU + Windows | 40.3 (0.260) | about 29,400 (190) |

  Computed from the list above (OCPU rate x OCPU + memory rate x 8 + Windows rate x OCPU); boot volume extra
  (about 1,000 JPY a month at an assumed 256 GB; the Windows minimum boot volume was not verified).
- **Free trial.** 300 USD of credits valid for up to 30 days, usable on any eligible service; the Japanese
  documentation says the same ("$300 … 最大30日間"), with no JPY equivalent. A phone number and a credit card are needed,
  the card is not charged unless you upgrade, and paid resources created with trial credits are reclaimed at the end
  unless the account is upgraded. The home region is chosen at sign-up. [official](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm),
  [official](https://docs.oracle.com/ja-jp/iaas/Content/FreeTier/freetier.htm)
- **Fit.** A 2 OCPU E4 Windows VM running 30 days costs about 180 USD of the 300 USD. Whether trial accounts may
  launch Windows images, and their E4/E5 service limits, was not confirmed.

## 5. Ranked recommendation (one-month bulk history plus occasional top-ups)

Decide by a one-day test, not a belief. The test on any candidate: Japanese language, display and system locale set;
install JV-Link (64-bit, with 32-bit as fallback) and set the key; `JVInit` returns 0; a small `RACE` setup
completes with the dialog answered by hand; one `JVRTOpen` of a past race returns time-series odds; the JVLinkAgent
service survives a reboot and a disconnected RDP session. Fail any step and move down the list.

| Rank | Option | Why | Main risk |
| --- | --- | --- | --- |
| 1 | Oracle Windows Server 2022, Tokyo or Osaka region, 2 OCPU E4/E5 Flex, 8 GB, on the trial credits | Near-zero cost (about 180 USD of the 300 USD), x86, matches #196, Japan region satisfies Art. 6, no meaningful lock-in | Server OS and cloud are both unsupported; trial reclaims paid resources after 30 days (upgrade or copy files first); English image needs a Japanese pack; unverified whether trial accounts can launch Windows. |
| 2 | Windows 365 Business Standard (Windows 11, 2 vCPU/8 GB), Japan East if selectable | Only option on the supported OS family (Windows 11, Japanese); about 6,746 JPY a month excl. tax for a month or two | Needs a Microsoft Entra tenant and a working order path from Japan; region choice and exact billing unconfirmed; still "cloud, unverified" for JRA-VAN. |
| 3 | ConoHa for Windows Server, 8 GB hourly (cap 9,746 JPY plus 1,386 RDS licence) | Hourly billing capped monthly; a user reported past time-series odds fetched this way; Japanese provider | Sign-ups paused when read; that report is paywalled and unnamed software; Windows Server again. |
| 4 | sakura VPS W8G or ABLENET Win3-4 on their trials (2 weeks / 10 days) | Cheap (3,490 to 10,780 JPY), trials let you run the day-1 test free | Trials are shorter than the bulk run; first-term prices rise on renewal; Windows Server again; no cloud reports. |
| 5 | Azure VM or Azure Virtual Desktop, Japan East | Real Windows 11 host possible; precise hourly prices | Licence entitlement and set-up are the most complex for a one-person month; costs more than 2-4. |
| 6 | Amazon WorkSpaces | None for this job | Licensed bundles are Windows Server; Windows 11 BYOL needs 50 desktops per Region. Not recommended. |

Plan-level risks, independent of the host:

1. **The data must be deleted after the free month or cancellation** (staff, quoted in section 2). If the files
   need to outlive the first month, keep the paid subscription (2,090 JPY a month, [official](https://jra-van.jp/dlb/)) for as long as they are used,
   and keep them private (Art. 7). Models and derived outputs that cannot restore the raw data are exempt unless distributed.
2. **Time-series odds history is "about one year, no guarantee"**: today's longer history may be removed. Export it
   in the first month and treat it as a snapshot.
3. **Free-month setup quirks**: the start-kit dialog must be answered by hand within about 5 seconds, so the first setup is
   an interactive RDP run, not a script. Unattended top-ups after that are less certain (no staff answer yet).
4. Keep the key on at most two machines; a cloud VM and the local UTM VM would use both.

## Not verified

- Whether JV-Link (installer, service key check, JVLinkAgent) runs on Windows Server 2019/2022 at all, on any provider;
  I found no first-hand success report on Oracle, AWS, Azure, ConoHa, sakura, ABLENET or Onamae.
- The original wording and date of the Windows Server exclusion (second-hand from a blog quoting the former BBS).
- Whether JRA-VAN blocks data-centre IP ranges; none was reported either way.
- Staff reply to the question whether JV-Link itself may run in a cloud Windows environment (unanswered when read),
  and the staff reply on wait times for many past 0B41/0B42 fetches (not visible).
- Whether an English Windows Server image with a Japanese language pack and Japanese system locale is accepted by
  JV-Link; whether IE Enhanced Security Configuration or other Server defaults block the installer or login page.
- Windows minimum boot volume on OCI and the trial account's service limits for E4/E5 and Windows images; Oracle's
  Windows 11 BYOL/dedicated-host pricing and whether an individual may use it.
- Windows 365 Business availability to a Japan buyer, region choice, and tax-included price; Azure Virtual Desktop
  entitlement costs; Onamae.com Desktop Cloud current plans and prices (a 2024 notice showed 2,640 JPY a month for
  2 GB, not confirmed on a current page); KAGOYA OS versions and RDS licence price; ConoHa sign-up status.
- The Google Cloud blog report (404 on fetch) and the contents of the paywalled ConoHa note.
- Performance on a cloud VM: all download-time figures are from other users' PCs.
