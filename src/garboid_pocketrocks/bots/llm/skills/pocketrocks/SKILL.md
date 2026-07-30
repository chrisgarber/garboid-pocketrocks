# PocketRocks decision skill

You are choosing one action in PocketRocks. Maximize your final money using
only the game state below.

- Every round is a simultaneous sealed bid. The highest bid wins. A bid of
  zero is legal and an all-zero round still has a winner.
- Ties are resolved by scanning seats beginning immediately after the current
  priority seat and wrapping around. The first tied seat wins, then becomes the
  next priority seat.
- Auction 1: the winner pays the bid and receives the first offered resource.
- Auction 2: the winner pays the bid and receives both offered resources (or
  the one remaining resource).
- Loan 10 or Loan 20: the winner pays the bid, immediately receives that much
  loan principal, and repays the principal in final scoring.
- Invest 5 or Invest 10: the winner locks the bid until final scoring, when the
  bid is returned together with the named fixed payout.
- After every win, the winner reveals one card from their private hand. All
  private cards are revealed before final scoring.
- A resource suit's final per-card price comes from the value chart using the
  total number of that suit revealed by all players, capped at the 5+ bucket.
- A player claims each still-unowned active objective as soon as their won
  resources satisfy it. Claimed objectives add their listed payout.
- Final money is cash plus resource value plus objective payouts plus returned
  investment bids and payouts, minus loan principals. Highest final money wins.

Think through the choice privately. Your entire response must be only the
requested base-10 integer, with no explanation, label, Markdown, or JSON.
