/*
 Navicat Premium Data Transfer

 Source Server         : localhost_3306
 Source Server Type    : MySQL
 Source Server Version : 50722
 Source Host           : localhost:3306
 Source Schema         : teamwork

 Target Server Type    : MySQL
 Target Server Version : 50722
 File Encoding         : 65001

 Date: 26/04/2021 20:49:58
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for t_board
-- ----------------------------
DROP TABLE IF EXISTS `t_board`;
CREATE TABLE `t_board`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `team_id` int(11) NULL DEFAULT NULL COMMENT '团队id',
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '描述',
  `user_id` int(11) NULL DEFAULT NULL COMMENT '创建用户id',
  `create_time` datetime NULL DEFAULT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '看板名称',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 9 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of t_board
-- ----------------------------
INSERT INTO `t_board` VALUES (8, 11, 'test-boardtest-boardtest-board', 14, '2021-04-25 12:16:17', 'test-board');

-- ----------------------------
-- Table structure for t_board_list
-- ----------------------------
DROP TABLE IF EXISTS `t_board_list`;
CREATE TABLE `t_board_list`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `board_id` int(11) NULL DEFAULT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT 'list名称',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `type` int(1) NULL DEFAULT NULL COMMENT '0 系统自带，1 用户创建',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 21 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of t_board_list
-- ----------------------------
INSERT INTO `t_board_list` VALUES (19, 8, 'card1', '2021-04-25 12:16:17', 1);

-- ----------------------------
-- Table structure for t_board_list_card
-- ----------------------------
DROP TABLE IF EXISTS `t_board_list_card`;
CREATE TABLE `t_board_list_card`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `list_id` int(11) NULL DEFAULT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '名称',
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '描述',
  `due_time` datetime NULL DEFAULT NULL COMMENT '到期日期',
  `attachments` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '附件',
  `attachments_status` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '状态',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `start_time` datetime NULL DEFAULT NULL,
  `end_time` datetime NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 27 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of t_board_list_card
-- ----------------------------
INSERT INTO `t_board_list_card` VALUES (22, 19, 'card2', 'test-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-card', NULL, NULL, NULL, '2021-04-25 12:16:33', '2021-04-25 04:16:29', '2021-04-25 04:16:31');
INSERT INTO `t_board_list_card` VALUES (23, 20, 'card3', 'test-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-card', NULL, NULL, NULL, '2021-04-25 12:17:11', '2021-04-25 04:17:07', '2021-05-05 16:00:00');
INSERT INTO `t_board_list_card` VALUES (24, 20, 'card4', 'test-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-cardtest-card', NULL, NULL, NULL, '2021-04-25 12:18:00', '2021-04-25 04:17:57', '2021-04-25 04:17:59');
INSERT INTO `t_board_list_card` VALUES (26, 20, '323232', NULL, NULL, NULL, NULL, '2021-04-25 21:11:55', '2021-04-22 16:00:00', '2021-04-29 16:00:00');

-- ----------------------------
-- Table structure for t_dictionary
-- ----------------------------
DROP TABLE IF EXISTS `t_dictionary`;
CREATE TABLE `t_dictionary`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `code` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
  `describe` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_bin ROW_FORMAT = COMPACT;

-- ----------------------------
-- Records of t_dictionary
-- ----------------------------
INSERT INTO `t_dictionary` VALUES (2, '02', '其他', '其他', '2021-04-18 11:12:01');

-- ----------------------------
-- Table structure for t_files
-- ----------------------------
DROP TABLE IF EXISTS `t_files`;
CREATE TABLE `t_files`  (
  `id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `card_id` int(11) NULL DEFAULT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '文件名称',
  `path` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of t_files
-- ----------------------------
INSERT INTO `t_files` VALUES ('94168480cab74f7a98f12b66048d9e39', 23, 'CPT202模版.pdf', 'CPT202模版.pdf', '2021-04-25 04:17:19');
INSERT INTO `t_files` VALUES ('b426d099d86141e88983ef72cfa44e72', 23, 'dropdown.js', 'dropdown.js', '2021-04-25 04:43:28');

-- ----------------------------
-- Table structure for t_team
-- ----------------------------
DROP TABLE IF EXISTS `t_team`;
CREATE TABLE `t_team`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '团队名称',
  `type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '类型',
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '描述',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `user_id` int(11) NULL DEFAULT NULL COMMENT '创建用户',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 12 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of t_team
-- ----------------------------
INSERT INTO `t_team` VALUES (11, 'team-test3', '1', 'team-test3team-test3team-test3team-test3', '2021-04-25 04:15:59', 14);

-- ----------------------------
-- Table structure for t_team_user
-- ----------------------------
DROP TABLE IF EXISTS `t_team_user`;
CREATE TABLE `t_team_user`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NULL DEFAULT NULL,
  `team_id` int(11) NULL DEFAULT NULL,
  `type` int(1) NULL DEFAULT NULL COMMENT '类型',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 10 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of t_team_user
-- ----------------------------
INSERT INTO `t_team_user` VALUES (9, 12, 11, NULL);

-- ----------------------------
-- Table structure for t_user
-- ----------------------------
DROP TABLE IF EXISTS `t_user`;
CREATE TABLE `t_user`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
  `username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
  `create_time` timestamp NULL DEFAULT NULL,
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `sex` int(1) NULL DEFAULT NULL COMMENT '性别 0男1女',
  `age` int(11) NULL DEFAULT NULL COMMENT '年龄',
  `phone` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL COMMENT '电话',
  `bio` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL COMMENT '介绍',
  `last_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近登录时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 15 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_bin ROW_FORMAT = COMPACT;

-- ----------------------------
-- Records of t_user
-- ----------------------------
INSERT INTO `t_user` VALUES (1, '管理员', 'admin', '1', '2021-04-18 11:12:01', '2021-04-18 11:12:01', 1, 22, '1111', NULL, '2021-04-25 12:11:10');
INSERT INTO `t_user` VALUES (12, 'test2', 'test2', 'test2', '2021-04-25 04:02:33', '2021-04-25 04:02:33', NULL, NULL, NULL, 'test2test2test2test2', '2021-04-26 12:39:22');
INSERT INTO `t_user` VALUES (14, 'test3', 'test3', 'test3', '2021-04-25 04:15:26', '2021-04-25 04:15:26', NULL, NULL, NULL, 'test3test3test3test3test3', '2021-04-25 04:15:32');

SET FOREIGN_KEY_CHECKS = 1;
