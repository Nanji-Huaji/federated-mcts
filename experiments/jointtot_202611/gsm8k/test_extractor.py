import unittest

from extractor import extract, reference


class FrozenPolicyTests(unittest.TestCase):
    def check(self, text, expected_status, expected_value):
        r = extract(text)
        self.assertEqual((r["status"], r["value"]), (expected_status, expected_value), text)

    def test_exact(self):
        for txt, n in [("#### **$1,250.00**", "1250"), ("#### −.5", "-1/2"),
                       ("#### 0.00000000000001", "1/100000000000000"),
                       (r"Final answer is \boxed{\frac{2}{4}}", "1/2"),
                       ("#### 5E-3", "1/200"), ("#### 18%", "18"), ("#### 0", "0")]:
            with self.subTest(txt=txt):
                self.check(txt, "formal", n)
        self.assertNotEqual(extract("#### 1000.01")["value"], reference("work\n#### 1000"))

    def test_priority(self):
        self.check("There were 99.\n#### 7\nVerification uses 43 items.", "formal", "7")
        self.check("#### 8\n99", "formal", "8")
        self.check("Answer: here is the derivation\n#### 5", "formal", "5")
        self.check("computed 2 cases. The final answer is: $9.50 dollars.", "answer_phrase", "19/2")
        self.check("Answer: #### 8", "formal", "8")
        self.check("Answer:\n\n 12", "answer_phrase", "12")
        self.check("Work took 12 steps.\n($3.50)", "terminal_scalar", "7/2")

    def test_equal_repeats_and_conflict(self):
        self.check("#### .5\nThe answer is 1/2", "formal", "1/2")
        for txt in ["#### 4\n#### 5", "#### 5\nfinal answer: 7", "Answer: 4 or 14",
                    r"\boxed{5}, or \boxed{6}", "#### 3 to 5"]:
            with self.subTest(txt=txt):
                self.check(txt, "conflict", None)

    def test_no_guessing(self):
        for txt in ["", "Calculate the result after 7 days", "3 + 4 = 7", "3 apples\n4 pears", "Step 2:", "1,34", "7/0"]:
            with self.subTest(txt=txt):
                self.check(txt, "no_answer", None)
        for txt in ["#### apples, 4", "Answer: 2+2=4", "#### 2 apples + 3 pears", "#### five\n5", "#### 1,00.5", "#### 2.2.2", "#### N/A"]:
            with self.subTest(txt=txt):
                self.check(txt, "malformed_declaration", None)

    def test_reference(self):
        self.assertEqual(reference("a lot of work with 1, 2 and 3.\n#### 18,000"), "18000")
        for txt in ["work ends at 4", "#### 3\n#### 4", "#### 3 or 4", "#### 4\nmore text"]:
            with self.subTest(txt=txt), self.assertRaises(ValueError):
                reference(txt)


if __name__ == "__main__":
    unittest.main()
