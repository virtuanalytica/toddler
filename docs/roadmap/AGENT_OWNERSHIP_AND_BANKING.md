# Agent ownership, legal vehicles and banking — design decision

**Decision date:** 11 October 2026. This is a product and pilot design, not an incorporation instruction or an assertion that an AI agent is a legal person. Bank pricing and eligibility must be rechecked before contracting.

## Ownership contract

The customer controls the economic unit: the agent's role, configuration, customer data, accepted work product, budget and proceeds, subject to third-party model licences. Toddler retains its own platform code and licenses the runtime, verification and routing service. The customer can export agent state and audit records in documented formats. A created profile without accepted paid work is not a paid agent.

| Pilot lane | Contracting party and account holder | Permitted agent authority | Transition gate |
| --- | --- | --- | --- |
| Low-risk private | Customer's existing sole proprietorship, if appropriate | Draft, analyse and propose within a small revocable budget | Move activity to a B.V. or other fitting legal person before significant credit, contract, investment or physical exposure |
| Higher-risk commercial | Customer-controlled B.V. or fitting legal person | Written mandate, limits, human approval, insurance and incident process | A separate B.V. per agent only if a real liability, funding or investor boundary justifies its cost |
| Joint commercial | Coöperatie with real members and agreed liability regime | Shared work, member-level accounting and governance | Member votes, profit distribution and bank eligibility checked in statutes |
| Public body | Existing public authority or approved joint body | Advice and bounded execution under public mandate; competent official retains reserved decisions | Procurement, privacy, impact assessment, audit and appeal path before rollout |
| Public-interest collective | Formal vereniging or stichting if its purpose and governance fit | Ledger per agent under the entity's account | Vereniging for member control; stichting for board-led mission and no distributions to founders |

A Dutch sole proprietor is personally liable and can have one sole proprietorship with multiple activities and trade names. A B.V. generally places liability with the company, subject to exceptions such as director misconduct. A foundation can legally have **one** board member. One person may also control three legal entities that become board members where statutes permit legal-person directors, but that does not create independent oversight or three independent UBOs. KVK requires natural-person UBO identification and conflict-of-interest rules still apply. If a grant, certification or public customer requires independent people, the three controlled entities do not satisfy that substantive requirement.

Primary sources: [sole proprietorship](https://business.gov.nl/running-your-business/legal-forms-and-governance/sole-trader-or-sole-proprietorship-in-the-netherlands/), [legal personality and corporate directors](https://ondernemersplein.overheid.nl/bedrijfsvoering/rechtsvormen-en-organisatie/rechtspersoonlijkheid/), [foundation board](https://www.kvk.nl/wetten-en-regels/bestuurder-van-een-stichting-dit-zijn-de-regels-en-taken/), [foundation UBOs](https://www.kvk.nl/ubo/wie-zijn-de-ubos-van-je-organisatie-g2/), [conflicts](https://www.kvk.nl/wetten-en-regels/stemmen-in-bestuur-van-stichting-en-belangen-die-botsen/), [association](https://business.gov.nl/running-your-business/legal-forms-and-governance/association/), [cooperative](https://business.gov.nl/running-your-business/legal-forms-and-governance/cooperative/).

## Agent account architecture

1. **Every agent has an internal subledger** keyed by immutable agent ID and owner ID. Entries are double-entry, append-only, in euros and linked to accepted invoice, PSP reference, fee schedule and audit hash.
2. **The real bank or PSP account belongs to the KYC-checked owner.** The agent receives a scoped spending mandate, never independent legal title to funds. Owner revocation takes effect before the next outbound payment.
3. **Dedicated IBAN is optional and provider-gated.** Pilot with a small number of agent-specific IBANs to simplify incoming payment references; at scale, use PSP virtual account identifiers or a common collection account with per-agent references if approved. Do not assume retail subaccounts can support 16,000 agents.
4. **No Toddler custody of customer funds** in an ordinary operating account. If Toddler facilitates buyer-to-owner settlement, use a licensed bank/payment institution or obtain the required authorisation; a stichting derdengelden alone does not remove PSD2 risk.
5. **Reconcile daily:** opening owner bank/PSP balance + settled inflows − payouts − fees = closing balance; sum of agent subledgers and owner suspense must match. Hold mismatches and disputed work for human resolution.
6. **Payment controls:** one-time approval for new payees, per-agent daily and monthly caps, two-person approval for high amounts, no autonomous borrowing or investments, and revocable API tokens. Public agents have zero independent payment authority by default.

De Nederlandsche Bank explains that a marketplace that itself receives buyer funds and later pays sellers generally needs a PSD2 licence or a licensed payment-service provider; merely routing through an affiliated foundation does not remove that issue. See [DNB's marketplace Q&A](https://www.dnb.nl/voor-de-sector/open-boek-toezicht/wet-regelgeving/psd2/elektronische-handelsplatformen-e-commerce-platforms-psd2/).

## Provider comparison for a Dutch pilot

| Service | Published offer checked 11 Oct 2026 | Agent fit | Unverified before production |
| --- | --- | --- | --- |
| [Knab Zakelijk](https://www.knab.nl/zakelijk/betalen/zakelijke-rekening-openen/zakelijke-rekeningen-vergelijken) | 5 payment accounts; sole proprietor €7/month or B.V. €11/month after promotional first year; 500 annual transfers included | Lowest-complexity manual pilot with at most a few agent IBANs | Automated creation, agent-specific mandates, B.V. eligibility of complex group structures |
| [bunq Pro Business](https://help.bunq.com/nl-nl/articles/what-plans-are-available) | €13.99/month, 25 accounts with unique IBANs; additional block of 25 at €20/month | Best published account density for a small portfolio; [API](https://doc.bunq.com/) merits sandbox proof | Provider acceptance of automated portfolios, per-owner limits, 16,000-agent scale and customer-funds flow |
| [Revolut Business Grow](https://www.revolut.com/en-NL/business/business-account-plans/) | €35/month; [Business API](https://help.revolut.com/nl-NL/help/integrating-with-external-apps/revolut-business-api/question-using-revolut-business-api/business/) starts at Grow; a [new same-currency account can have a new IBAN](https://help.revolut.com/nl-NL/help/receiving-payments/transfers-info/international-iban/business/) | Good candidate for controlled payouts, FX and webhooks | Bulk account creation, subaccount limits, contract eligibility and fees under realistic volume |
| [N26 Business](https://support.n26.com/en-de/memberships-and-account-types/business-accounts/how-does-n26-business-work) | Freelancer in personal name; states B2B payments are unsupported | Unsuitable for the proposed B2B agent marketplace | None needed for pilot selection |

**Pilot choice:** start with a regulated PSP marketplace flow and one owner-level bank account. bunq Pro is the first small-portfolio IBAN experiment if its terms permit it; Knab is the simpler 5-account comparison; Revolut Grow is the API/payout comparison. The experiment should record exact fees, KYC time, reconciliation coverage, exceptions and revocation latency. No live account has been opened by this design.

## Public-sector lane

A government can retain an agent within its existing legal body, budget and bank account. It need not form a vereniging. Choose a formal vereniging only when genuine members must jointly control the mission and surpluses are reinvested. Choose a stichting for a board-governed public-interest mission without members; a coöperatie for members sharing commercial benefits; or a public joint body when government powers or intergovernmental accountability call for it. Procurement and competence rules depend on the specific activity. The Dutch data protection authority identifies extra impact-assessment and oversight duties for some high-risk public AI systems: [AP guidance](https://autoriteitpersoonsgegevens.nl/actueel/de-fria-voor-ai-systemen-komt-eraan-bereid-u-voor).

## Evidence required before scale

- Contract names actual owner, beneficial owners, data/IP rights, mandate, payout party, liability, insurance and exit procedure.
- Payment-provider design approved for the exact marketplace flow and client-funds boundary.
- For at least two paid owners and 25 active agent-months: zero unexplained ledger variance, 100% bank/PSP reconciliation, no payment outside mandate, and measured net owner proceeds and platform contribution.
- For a public pilot: published mandate, procurement route, privacy and fundamental-rights assessments where required, human appeal, and independent audit access.
