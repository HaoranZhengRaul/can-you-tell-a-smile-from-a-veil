# Blind labelling app: the human blind read (Session 9)

People read the same 100 listings the two LLM readers labelled in Session 8 (25 written before ChatGPT, 75 from
2025 to 2026, shuffled) and say, for each: **AI wrote all or part of it**, **Not sure**, or **A person wrote it**.
Their labels give the detector the human check the DETECTOR RULE asks for (iii), saved before anyone opens the key.

- `template.html`: the app (one HTML file, no outside scripts, works on phones). It shows each rater the listings in
  their own random order (seeded by their ID, so the order can be rebuilt), with two attention checks, keyboard
  answers (1, 2, 3; Backspace goes back), saved progress if the page reloads, and response times.
- `build_label_app.py`: fills the template. It scrubs contact details (e-mail, phone numbers, links, a name after
  "contact", "call", "ask for"), gives every listing a new random id, adds the attention checks, and writes the
  app and, separately, the private files. `--selftest` runs its tests; `--demo` builds six invented listings.

The builder never reads the key (`results/blind_s8/KEY_do_not_open.csv`). Nothing in the hosted file links a
listing to its blind id: that map is `private/id_map.csv`, which is never hosted or committed.

## Steps

1. **Get the texts** (Mercury, when port 22 answers). The file holds raw listing text, so it goes to a scratch
   folder outside this repository, never into git:
   ```
   scp mercury:/project/dmv_project/projects/ai_pricing_kbb/results/blind_s8/texts.jsonl SCRATCH/blind_texts.jsonl
   ```
2. **Consent text.** Write the consent text the IRB approves as a small HTML file (`consent.html`). Ask the
   UChicago IRB whether the study needs review before it launches. A draft to adapt:
   ```html
   <p>This is a research task from the University of Chicago. You will read 100 short listings from a government
   auction website and say whether you think an AI writing tool wrote all or part of each one. It takes about 20
   minutes. We record your answers, how long each took, and your platform ID; nothing else about you. Taking
   part is voluntary and you can stop at any time. Questions: [contact for the study].</p>
   ```
3. **A place for the answers** (recommended): a Google Sheet with this Apps Script (Extensions, then Apps Script;
   paste, save, then Deploy, New deployment, Web app, Execute as "Me", Who has access "Anyone"; copy the URL):
   ```javascript
   function doPost(e) {
     var lock = LockService.getScriptLock();
     lock.waitLock(20000);
     try {
       var ss = SpreadsheetApp.getActiveSpreadsheet();
       var sh = ss.getSheetByName("responses") || ss.insertSheet("responses");
       if (sh.getLastRow() === 0) {
         sh.appendRow(["received_at", "rater_id", "study_id", "session_id", "app_version", "item_set_hash",
                       "attention_passed", "ai_use", "n_responses", "payload_json"]);
       }
       var d = JSON.parse(e.postData.contents);
       sh.appendRow([new Date(), d.rater_id, d.study_id, d.session_id, d.app_version, d.item_set_hash,
                     d.attention_passed, d.ai_use, (d.responses || []).length, e.postData.contents]);
       return ContentService.createTextOutput("ok");
     } finally {
       lock.releaseLock();
     }
   }
   ```
   Without it, the app still works: at the end it offers each rater a download and a one-line copy of their
   answers to send in.
4. **Build** (prints the item-set hash that every answer carries):
   ```
   python3 code/label_app/build_label_app.py --texts SCRATCH/blind_texts.jsonl --expect-sha256 73e9a2c8 \
     --consent consent.html --submit-url "APPS_SCRIPT_URL" \
     --completion-url "COMPLETION_URL_FROM_PROLIFIC" --completion-code "CODE" \
     --out output/label_app --private SCRATCH/label_app_private
   ```
   `output/` is ignored by git. Read `SCRATCH/label_app_private/scrub_report.json` before hosting: it lists every
   change and every number the scrubber left alone that might be a phone number.
5. **Host** `output/label_app/index.html` on any static host (Netlify Drop, GitHub Pages in a separate repository,
   a university web space). Whatever host you use, the scrubbed listing texts become public there.
6. **Recruit.** On Prolific, choose URL parameters; the study URL is
   `https://YOUR_HOST/index.html?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`.
   Other platforms work too: the app also reads `workerId`, `participant`, `pid` or `rater`. Test it yourself first
   (`.../index.html?PROLIFIC_PID=TEST`) and check that a row reaches the Sheet. Every rater labels all 100; about 10
   raters leave room to drop those who fail an attention check and still give five or more labels per listing.
7. **Return the answers.** Download the Sheet as CSV (or collect the downloaded JSON files) into the session
   scratchpad and say so. The analysis joins them with `id_map.csv` and only then with the key on Mercury.

## What each answer holds

One record per rater: `rater_id`, `study_id`, `session_id`, `app_version`, `item_set_hash`, `consent_at`,
`started_at`, `finished_at`, `order` (the ids in the order shown), `responses` (per id: `label` A, U or H,
`rt_ms`, `revisions`), `attention` (expected and given answer for each check), `attention_passed`, `ai_use` and
`comment`.

## What the raters are told

The instructions say what the LLM readers' brief said, without its example tells: listings come from a government
auction site whose 2025 button rewrites listings with AI; 25 of the 100 were written before AI writing tools existed
and 75 come from 2025 or 2026, some written or rewritten by AI; do not paste the text into any tool. Leaving out
the tells keeps the human labels independent of the detector's marker words.
