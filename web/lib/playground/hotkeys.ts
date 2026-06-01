/** True when the user is typing in a field — playground shortcuts should not fire. */
export function isTypingInteractionTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  if (target.closest('[role="dialog"]')) return true;
  return false;
}
