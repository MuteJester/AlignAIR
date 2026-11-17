/**
 * The CLI | Python tab every example shares, remembered across pages, and the text a copy button copies. Pure, so
 * `npm test` checks it.
 */
export type Tab = "cli" | "python";
export const TAB_KEY = "alignair.codeTab";

interface Store {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function readTab(store: Store | null): Tab {
  try {
    const value = store?.getItem(TAB_KEY);
    return value === "python" ? "python" : "cli";
  } catch {
    return "cli";
  }
}

export function writeTab(store: Store | null, tab: Tab): void {
  try {
    store?.setItem(TAB_KEY, tab);
  } catch {
    /* storage blocked (private mode): the choice lasts for this page only */
  }
}

/**
 * What a copy button copies: for a transcript (lines starting `$ `), the commands without the prompt, with their
 * continuation lines, and no output; any other code as it is.
 */
export function commandText(code: string): string {
  const lines = code.split("\n");
  if (!lines.some((line) => line.startsWith("$ "))) return code;
  const out: string[] = [];
  let continuing = false;
  for (const line of lines) {
    if (line.startsWith("$ ")) {
      out.push(line.slice(2));
      continuing = line.endsWith("\\");
    } else if (continuing) {
      out.push(line);
      continuing = line.endsWith("\\");
    }
  }
  return out.join("\n");
}
