"""Blank Street palette and type for My Day."""

from __future__ import annotations

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,650&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background:
    radial-gradient(920px 440px at 50% -90px, rgba(255, 252, 248, 0.96), rgba(249, 240, 233, 0) 64%),
    linear-gradient(180deg, #FBF7F2 0%, #F3E6DA 100%) !important;
  color: #44341D;
  font-family: "Outfit", "Segoe UI", "Helvetica Neue", sans-serif;
  font-size: 22px;
}
header[data-testid="stHeader"] {
  background: #F9F0E9 !important;
  height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
#MainMenu,
footer {
  display: none !important;
}
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
.block-container {
  position: relative;
  max-width: 760px;
  padding-top: 1.4rem;
  padding-bottom: 3rem;
}
h1 {
  font-family: "Outfit", "Segoe UI", sans-serif !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  letter-spacing: 0.22em !important;
  text-transform: uppercase !important;
  text-align: center !important;
  color: #44341D !important;
  line-height: 1.2 !important;
  margin: 0 0 0.15rem !important;
}
h1 a { color: #44341D !important; text-decoration: none !important; }
.tagline {
  margin: 0 0 0.85rem;
  text-align: center;
  font-size: 22px;
  line-height: 1.35;
  color: #44341D;
}
[data-testid="stMarkdownContainer"] p.greeting {
  font-family: "Fraunces", Georgia, "Times New Roman", serif !important;
  font-size: 52px !important;
  font-weight: 650 !important;
  letter-spacing: -0.03em !important;
  line-height: 1.25 !important;
  color: #44341D !important;
  margin: 0.7rem 0 0.9rem !important;
}
[data-testid="stMarkdownContainer"] p.day-part {
  margin: 1.2rem 0 0.1rem !important;
  font-size: 30px !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
  color: #44341D !important;
}
[data-testid="stMarkdownContainer"] p.quiet-note,
[data-testid="stMarkdownContainer"] p.llm-line,
[data-testid="stMarkdownContainer"] p.clip-source,
[data-testid="stMarkdownContainer"] p.tagline {
  margin: 0.15rem 0 0.35rem !important;
  color: #44341D !important;
  font-size: 22px !important;
  line-height: 1.4 !important;
}
[data-testid="stMarkdownContainer"] p.tagline {
  text-align: center !important;
  margin-bottom: 0.85rem !important;
}
[data-testid="stMarkdownContainer"] p.llm-line { font-weight: 500 !important; }
[data-testid="stMarkdownContainer"] p.clip-time {
  display: inline-block !important;
  margin: 0.7rem 0 0.35rem !important;
  padding: 0.28rem 0.85rem !important;
  border-radius: 999px !important;
  background: #ECE0D4 !important;
  color: #44341D !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
}
[data-testid="stMarkdownContainer"] p.clip-line,
[data-testid="stMarkdownContainer"] p.answer-line {
  margin: 0.15rem 0 0.35rem !important;
  color: #44341D !important;
  font-size: 22px !important;
  line-height: 1.45 !important;
}
[data-testid="stMarkdownContainer"] p.answer-line { font-size: 26px !important; }
[class*="st-key-quiet-eval"] {
  position: absolute;
  top: 0.15rem;
  right: 0.15rem;
  z-index: 20;
  width: 7.2rem;
}
button[data-testid^="stBaseButton-"] {
  min-height: 56px !important;
  border-radius: 999px !important;
  font-family: "Outfit", "Segoe UI", sans-serif !important;
  font-size: 22px !important;
  font-weight: 600 !important;
  padding: 0.4rem 1rem !important;
  box-shadow: none !important;
}
button[data-testid="stBaseButton-primary"] {
  background: #52735E !important;
  color: #FDF8F5 !important;
  border: 1.5px solid #52735E !important;
}
button[data-testid="stBaseButton-primary"] p,
button[data-testid="stBaseButton-primary"] span,
button[data-testid="stBaseButton-primary"] div {
  color: #FDF8F5 !important;
}
button[data-testid="stBaseButton-secondary"] {
  background: rgba(253, 248, 245, 0.72) !important;
  color: #44341D !important;
  border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
}
button[data-testid="stBaseButton-secondary"] p,
button[data-testid="stBaseButton-secondary"] span {
  color: #44341D !important;
}
button[data-testid="stBaseButton-tertiary"] {
  background: transparent !important;
  color: #44341D !important;
  border: none !important;
  font-size: 20px !important;
}
button[data-testid="stBaseButton-tertiary"] p,
button[data-testid="stBaseButton-tertiary"] span {
  color: #44341D !important;
}
button[data-testid^="stBaseButton-"]:disabled {
  opacity: 0.55;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stMarkdownContainer"] p {
  color: #44341D;
}
[data-testid="stWidgetLabel"] p {
  font-size: 22px !important;
  font-weight: 600 !important;
}
[data-testid="stTextInput"] input {
  min-height: 56px;
  border-radius: 18px;
  background: rgba(253, 248, 245, 0.78);
  color: #44341D;
  border: 1.5px solid rgba(255, 255, 255, 0.8);
  font-size: 22px !important;
}
[data-testid="stChatMessage"] {
  background: rgba(253, 248, 245, 0.62);
  border: 1px solid rgba(255, 255, 255, 0.8);
  border-radius: 24px;
  padding: 0.8rem 1rem;
  box-shadow: 0 10px 28px rgba(68, 52, 29, 0.06);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: #ECE0D4;
  width: 88%;
  margin-left: auto;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  background: #52735E !important;
  color: #FDF8F5 !important;
}
[data-testid="stMarkdownContainer"] p.chat-question {
  margin: 0 !important;
  font-size: 22px !important;
  font-weight: 600 !important;
  color: #44341D !important;
}
[data-testid="stMarkdownContainer"] p.result-head {
  margin: 0.9rem 0 0.1rem !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  color: #44341D !important;
}
[data-testid="stChatInput"] {
  border-radius: 24px !important;
  background: rgba(253, 248, 245, 0.95) !important;
  border: 1.5px solid rgba(228, 213, 196, 0.95) !important;
}
[data-testid="stChatInput"] textarea {
  font-size: 22px !important;
  color: #44341D !important;
  min-height: 56px !important;
}
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
  background: #F3E6DA !important;
}
[class*="st-key-shown-row-"] {
  padding: 0.4rem !important;
  border-radius: 20px !important;
}
[class*="st-key-shown-row-"] [data-testid="stImage"] img {
  width: 100% !important;
  height: 96px !important;
  object-fit: cover !important;
  border-radius: 14px !important;
}
[class*="st-key-shown-row-"] .thumb-missing {
  height: 96px;
  border-radius: 14px;
  background: #ECE0D4;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
}
.row-words { color: #44341D; line-height: 1.35; padding-bottom: 1rem; }
.row-words .row-time {
  display: inline-block;
  margin-bottom: 0.3rem;
  padding: 0.12rem 0.6rem;
  border-radius: 999px;
  background: #ECE0D4;
  font-size: 14px;
  font-weight: 700;
}
.row-words .row-line { font-size: 18px; margin-bottom: 0.25rem; }
.row-words .row-meta { font-size: 15px; margin-bottom: 0.15rem; }
.row-words .row-source {
  font-size: 14px;
  opacity: 0.8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
[class*="st-key-shown-row-"] button[data-testid^="stBaseButton-"] {
  min-height: 42px !important;
  font-size: 17px !important;
  padding: 0.2rem 0.8rem !important;
}
[class*="st-key-shown-row-"] [data-testid="stVerticalBlock"] { gap: 0.3rem; }
[class*="st-key-ask-suggestion"] button {
  min-height: 48px !important;
  border-radius: 999px !important;
  background: rgba(253, 248, 245, 0.72) !important;
  border: 1.5px solid rgba(228, 213, 196, 0.95) !important;
  color: #44341D !important;
  font-size: 19px !important;
  padding: 0.3rem 1rem !important;
}
[class*="st-key-ask-suggestion"] button:hover {
  background: #ECE0D4 !important;
  border-color: #52735E !important;
}
[data-testid="stCaptionContainer"] p {
  color: #44341D !important;
  font-size: 20px !important;
}
[data-testid="stAlert"] {
  background: rgba(253, 248, 245, 0.72) !important;
  color: #44341D !important;
  border: 1px solid rgba(255, 255, 255, 0.8) !important;
  border-radius: 20px !important;
  backdrop-filter: blur(16px);
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
  color: #44341D !important;
  font-size: 22px !important;
}
[class*="st-key-feed-"],
[class*="st-key-add-moment"],
[class*="st-key-shown-"],
[class*="st-key-thumb-slot-"] {
  background: rgba(253, 248, 245, 0.58) !important;
  border: 1px solid rgba(255, 255, 255, 0.78) !important;
  border-radius: 28px !important;
  padding: 0.55rem !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.88),
    0 16px 36px rgba(68, 52, 29, 0.08);
  backdrop-filter: blur(20px) saturate(150%);
  -webkit-backdrop-filter: blur(20px) saturate(150%);
}
[class*="st-key-gallery"] {
  gap: 16px !important;
}
[class*="st-key-gallery"] [data-testid="stHorizontalBlock"] {
  gap: 16px !important;
}
[class*="st-key-thumb-slot-"] {
  min-height: 248px;
}
[class*="st-key-thumb-slot-"] a {
  display: block;
  line-height: 0;
  border-radius: 20px;
}
[class*="st-key-thumb-slot-"] [data-testid="stImage"] {
  width: 100%;
}
[class*="st-key-thumb-slot-"] [data-testid="stImage"] img {
  width: 100% !important;
  height: 168px !important;
  object-fit: cover !important;
  border-radius: 20px !important;
}
[data-testid="stMarkdownContainer"] p.thumb-label {
  margin: 0.4rem 0 0.2rem !important;
  text-align: center !important;
  color: #44341D !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
}
[class*="st-key-thumb-slot-"] button,
[class*="st-key-gallery-back"] button,
[class*="st-key-gallery-next"] button,
[class*="st-key-clip-back"] button {
  min-height: 64px !important;
}
[data-testid="stVideo"],
[data-testid="stVideo"] video {
  border-radius: 22px;
  overflow: hidden;
  background: #ECE0D4;
}
[data-testid="stFileUploader"] section {
  background: rgba(253, 248, 245, 0.72);
  border-radius: 20px;
  border: 1.5px dashed rgba(228, 213, 196, 0.95);
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] small {
  display: none !important;
}
[data-testid="stFileUploader"] button {
  min-height: 56px !important;
  width: 100% !important;
  font-size: 22px !important;
}
[data-testid="stExpander"] details {
  background: rgba(253, 248, 245, 0.72) !important;
  border: 1px solid rgba(255, 255, 255, 0.85) !important;
  border-radius: 20px !important;
}
[data-testid="stExpander"] summary p {
  color: #44341D !important;
  font-size: 20px !important;
  font-weight: 600 !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
  font-size: 20px !important;
  line-height: 1.45 !important;
  margin: 0.1rem 0 !important;
}
[data-testid="stExpander"] [data-testid="stIconMaterial"] {
  color: #52735E !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
  color: #44341D !important;
}
[data-testid="stMetricValue"] { font-size: 32px !important; }
[data-testid="stMetricLabel"] { font-size: 18px !important; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.75rem; }
</style>
"""
