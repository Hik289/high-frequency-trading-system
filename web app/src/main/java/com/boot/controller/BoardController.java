package com.boot.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import com.boot.entity.Board;
import com.boot.entity.BoardList;
import com.boot.service.BoardService;
import com.boot.vo.BaseMessage;
import com.boot.vo.MessageHandler;
/**
 * board
 */
@Controller
@RequestMapping("/private/board")
public class BoardController {
	@Autowired
	private BoardService boardService;
	/**
	 * board
	 */
	@RequestMapping("/addBoard")
	public String addBoard(ModelMap model,Integer teamId) {
		model.put("teamId", teamId);
		return "page/team_add_board";
	}
	/**
	 * save
	 * @param board
	 * @return
	 */
	@ResponseBody
	@RequestMapping("/addBoard/save")
	public  BaseMessage<?> addBoardSave(Board board) {
		boardService.addBoardSave(board);
		return MessageHandler.createSuccessVo("operate successfully");
	}
	/**
	 * board list
	 */
	/**
	 * board list save
	 */
	@ResponseBody
	@RequestMapping("/list/save")
	public  BaseMessage<?> listSave(BoardList list) {
		boardService.listSave(list);
		return MessageHandler.createSuccessVo("operate successfully");
	}
}
