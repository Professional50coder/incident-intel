# Design notes

Patterns observed in two public product sites (navige.ai and proposify.com), viewed in a
browser in October 2026 as a visitor only. No sign-up, log-in or form entry was done, so
neither product's inside screens were seen: both gate the product behind an account. Only
patterns are recorded; none of their text, branding, imagery or code is used. Coverage was
partial: on navige.ai I scrolled the landing page and opened the pricing page (its docs path
returned a not-found page and I did not open About properly); on proposify.com I read the
page text structure, and screenshots were blocked by the browser tool.

## Patterns observed

| Area | Pattern |
|---|---|
| Navigation | Floating, rounded "pill" bar that stays at the top; 3-4 links, a quiet text link for sign-in and one filled primary button |
| Hero | One outcome-focused headline with a single emphasised phrase in a contrasting (serif italic) style; one sentence of support; a filled primary CTA and a quieter secondary CTA; a product-launch badge above the headline |
| Proof strip | A row of logos or numeric stats directly after the hero (both sites). Here only verifiable facts are allowed, so it becomes facts about the project (licence, tests, CI, benchmark number) |
| Section order | Hero, proof strip, 3-up value cards, large statement heading, feature sections with a visual each, how it works, testimonials or integrations, pricing or final CTA, footer |
| Feature sections | Alternating text and product visual, one benefit per section, one CTA per section |
| Cards | Generous radius, soft gradient or flat fill, one-line caption, thin divider above the caption |
| Typography | Very large tight sans headlines, medium body, scale changes in big steps; italics used sparingly for emphasis |
| Motion | Sections fade and rise in on scroll, marquee-style logo strips; hover lifts on cards |
| Pricing / final CTA | Short, human headline, one visual metaphor, one primary action |
| Onboarding | Public "start free" goes straight to sign-up; there is no anonymous demo, which is the main thing incident-intel does better (try a sample with no account) |

## Applied to incident-intel (original identity)

- Own palette: warm paper and deep ink, teal brand colour, coral reserved for flagged moments. Tokens live in `src/incident_intel/api/static/ds.css`; every page uses them (light and dark, with a manual toggle).
- Landing (`/`): floating pill nav; outcome headline with an italic emphasised word; "Try a sample" primary CTA; real product screenshot in a window frame; a proof strip of verifiable facts only (MIT licence, tests, CI, benchmark with its caveat, runs on CPU, no account); three alternating feature sections using real output (ranked review queue, one slider for the false-alarm budget, report export); how it works; who it is for; honest limits; FAQ; final CTA; footer.
- App (`/app`): app shell with a top bar and a sidebar (new analysis, reports history, guide) that collapses on phones; cards, KPI tiles, tables, badges, empty and loading states from the same component set; one primary action per screen.
- Not copied: any wording, logos, imagery, gradients or layout measurements. No testimonials, customer logos or usage statistics are shown because there are none to verify.
