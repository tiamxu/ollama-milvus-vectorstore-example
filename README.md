#
```
CREATE DATABASE `test` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci

 CREATE TABLE `qa_pairs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `question` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `answer` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=42 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci

```