/**
 * Arguments of the scripts of the package.
 */

/**
 * Removes the `--` that separates the arguments of a script from the ones of the package manager.
 *
 * The package manager forwards the separator to the script, and Node's `parseArgs` stops parsing
 * the options at it: every option written after it would be a positional. The separator is
 * removed, so that the options of a script work the same way with and without it.
 */
export function stripArgumentSeparator(): void {
	const args = process.argv.slice(2);
	const index = args.indexOf("--");
	if (index === -1) {
		return;
	}

	args.splice(index, 1);
	process.argv = [...process.argv.slice(0, 2), ...args];
}
