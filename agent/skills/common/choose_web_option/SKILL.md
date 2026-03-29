# Skill: Choose Web Option

Use when:
- The task is a no-visual web or online-service selection benchmark.
- The user needs one option chosen from an explicit candidate list for booking, ordering, routing, or place selection.

Supported action:
- `choose_web_option`

Parameters:
- `option_id`: candidate ID copied exactly from the provided list, such as `A`, `B`, or `C`
- `option_text`: exact candidate text copied from the provided list
- `service`: short service label such as `ticket_booking`, `food_delivery`, `hotel_booking`, or `map_route_selection`

Rules:
- Choose exactly one provided candidate. Do not invent a new option.
- Use dialogue context for temporary goals such as urgency, comfort, or current need.
- Use long-term memory for durable constraints such as fear of heights, diet, accessibility, or noise sensitivity.
- Prefer the option that satisfies both the immediate request and the durable preference.
- Copy `option_text` exactly so the evaluation runner can score the decision reliably.

Examples:
- Travel booking with fear-of-heights preference:
  - `action`: `choose_web_option`
  - `parameters`: `{"option_id": "B", "option_text": "high-speed rail ticket", "service": "ticket_booking"}`
- Food delivery with stomach sensitivity:
  - `action`: `choose_web_option`
  - `parameters`: `{"option_id": "A", "option_text": "plain congee", "service": "food_delivery"}`
