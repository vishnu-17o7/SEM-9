#!/usr/bin/env python3
import sys

from spam_utils import classify


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Usage:  python classify.py <text>")
        print("   or:  echo 'some text' | python classify.py")
        sys.exit(1)

    is_spam, prob = classify(text)
    label = "SPAM" if is_spam else "HAM"
    print(f"Prediction:      {label}")
    print(f"Spam confidence: {prob:.2%}")


if __name__ == "__main__":
    main()
